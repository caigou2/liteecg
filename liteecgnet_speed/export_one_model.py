#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import tensorflow as tf
import torch

from config import *
from models import BiRCNN, DeepECGNet, ECGNet, LDCNN, LiteECGNet, ResNet, SE_ECGNet


def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        if "total_params" in name or "total_ops" in name:
            print(f"  [Info] Skipping redundant key: {name}")
            continue
        new_state_dict[name] = v
    return new_state_dict


def load_checkpoint(weights_path):
    ckpt = torch.load(str(weights_path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def get_model_instance(model_name, input_len):
    if model_name == "ecgnet":
        return ECGNet(config=ECGNET_CONFIG)
    elif model_name == "se_ecgnet":
        return SE_ECGNet(config=SE_ECGNET_CONFIG)
    elif model_name == "bircnn":
        return BiRCNN(config=BIRCNN_CONFIG)
    elif model_name == "resnet":
        return ResNet(config=RESNET_CONFIG)
    elif model_name == "liteecgnet":
        return LiteECGNet(config=LITEECGNET_CONFIG)
    elif model_name == "deepecgnet":
        return DeepECGNet(config=DEEPECGNET_CONFIG)
    elif model_name == "ldcnn":
        return LDCNN(config=LDCNN_CONFIG, input_len=input_len)
    else:
        return None


def export_onnx(model, dummy_input, onnx_path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def find_saved_model_dir(root_dir: Path) -> Path:
    for p in root_dir.rglob("saved_model.pb"):
        return p.parent
    raise FileNotFoundError(f"No saved_model.pb found under: {root_dir}")


def make_representative_dataset(rep_npy: Path):
    rep_data = np.load(str(rep_npy), allow_pickle=False)

    if rep_data.ndim == 0:
        raise ValueError("--rep-npy must contain at least one sample")

    def representative_dataset():
        max_samples = min(len(rep_data), 100)
        for i in range(max_samples):
            sample = np.asarray(rep_data[i], dtype=np.float32)

            # ECG shapes:
            # [L] -> [1, 1, L]
            # [1, L] -> [1, 1, L]
            # [1, 1, L] -> keep
            if sample.ndim == 1:
                sample = sample.reshape(1, 1, -1)
            elif sample.ndim == 2:
                sample = sample.reshape(1, *sample.shape)
            elif sample.ndim == 3:
                pass
            else:
                raise ValueError(f"Unsupported representative sample shape: {sample.shape}")

            yield [sample]

    return representative_dataset


def onnx_to_tflite_with_onnx2tf(
    onnx_path: Path,
    tflite_path: Path,
    input_shape_str: str,
    quant: str = "none",
    rep_npy: Optional[Path] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir) / "onnx2tf_out"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Keep the input shape absolutely unchanged.
        # This is the key fix for the Tile layout problem on 1D ECG models.
        cmd = [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(work_dir),
            "-b",
            "1",
            "-ois",
            f"input:{input_shape_str}",
            "-kat",
            "input",
        ]

        subprocess.run(cmd, check=True)

        saved_model_dir = find_saved_model_dir(work_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

        if quant == "none":
            pass
        elif quant == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quant == "int8":
            if rep_npy is None:
                raise ValueError("INT8 quantization requires --rep-npy")

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = make_representative_dataset(rep_npy)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            raise ValueError(f"Unknown quant mode: {quant}")

        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)


def main():
    parser = argparse.ArgumentParser(
        description="Export one PyTorch ECG model to ONNX/TFLite using onnx2tf"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["liteecgnet", "ldcnn", "ecgnet", "se_ecgnet", "resnet", "deepecgnet", "bircnn"],
        help="Model name",
    )
    parser.add_argument("--weights", required=True, help="Path to the .pt file")
    parser.add_argument("--out-dir", default="./exports", help="Directory to save output files")
    parser.add_argument("--segment-sec", type=float, default=2.0, help="Segment length in seconds")
    parser.add_argument("--fs", type=int, default=360, help="Sampling frequency")
    parser.add_argument("--export-format", choices=["onnx", "tflite", "both"], default="both")
    parser.add_argument("--quant", choices=["none", "fp16", "int8"], default="none")
    parser.add_argument("--rep-npy", default=None, help="Representative dataset .npy for INT8")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    rep_npy = Path(args.rep_npy).resolve() if args.rep_npy else None

    input_len = int(round(args.segment_sec * args.fs))
    dummy_input = torch.randn((1, 1, input_len))
    input_shape_str = f"1,1,{input_len}"

    model = get_model_instance(args.model_name, input_len)
    if model is None:
        raise ValueError(f"Unknown model: {args.model_name}")

    state_dict = load_checkpoint(weights_path)
    cleaned_dict = clean_state_dict(state_dict)
    model.load_state_dict(cleaned_dict, strict=True)
    model.eval()

    onnx_path = out_dir / f"{args.model_name}.onnx"
    tflite_path = out_dir / f"{args.model_name}.tflite"

    if args.export_format in ("onnx", "both"):
        print(f"[1/2] Exporting ONNX: {onnx_path}")
        export_onnx(model, dummy_input, onnx_path)
        print("  Done.")

    if args.export_format in ("tflite", "both"):
        print(f"[2/2] Exporting TFLite: {tflite_path}")
        if not onnx_path.exists():
            export_onnx(model, dummy_input, onnx_path)

        onnx_to_tflite_with_onnx2tf(
            onnx_path=onnx_path,
            tflite_path=tflite_path,
            input_shape_str=input_shape_str,
            quant=args.quant,
            rep_npy=rep_npy,
        )
        print("  Done.")

    print("Finished.")


if __name__ == "__main__":
    main()
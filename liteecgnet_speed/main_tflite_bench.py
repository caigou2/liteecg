#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import numpy as np
import psutil
import resource 
from multiprocessing import Process, Queue

# 尝试载入 TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("错误: 请安装 tflite-runtime")
        exit(1)

# 功耗模型系数
def calculate_power_model(cpu_usage_pct):
    base_power = 600.0  
    scaler = 28.5  
    return round(base_power + (cpu_usage_pct * scaler), 0)

def worker_measure(tflite_path, samples, model_name, warmup, result_queue):
    """
    独立进程运行推理评测
    """
    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss / (1024 * 1024)

    try:
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        expected_shape = tuple(input_details[0]['shape']) # 例如 (1, 720, 1)

        # 1. 预热
        for i in range(min(warmup, len(samples))):
            x = samples[i].reshape(expected_shape)
            interpreter.set_tensor(input_index, x)
            interpreter.invoke()
        
        # 2. 核心测量循环
        psutil.cpu_percent(interval=None) 
        latencies = []
        
        for x_raw in samples:
            # 确保输入维度与模型一致
            x = x_raw.reshape(expected_shape)
            t_start = time.perf_counter()
            interpreter.set_tensor(input_index, x)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_index) # 模拟读取结果
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)
        
        avg_cpu_load = psutil.cpu_percent(interval=0.1)
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        
        avg_ms = np.mean(latencies)
        avg_power = calculate_power_model(avg_cpu_load)
        energy_per_sample = (avg_power * avg_ms) / 1000.0
        
        result_queue.put({
            "status": "success",
            "Model Name": model_name,
            "Average Time (ms)": round(avg_ms, 3),
            "Peak Memory (MB)": round(peak_mem, 2),
            "Power (mW)": int(avg_power),
            "Energy (mJ)": round(energy_per_sample, 4)
        })

    except Exception as e:
        result_queue.put({"status": "error", "Model Name": model_name, "message": str(e)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None, help="原始数据路径 (仅首次运行需要)")
    parser.add_argument('--model-dir', type=str, default="save_tflite", help="TFLite模型路径")
    parser.add_argument('--cache-file', type=str, default="test_samples_100.npy", help="100个样本的缓存文件")
    args = parser.parse_args()

    # --- 第一部分：获取这 100 个 NumPy 样本 ---
    if os.path.exists(args.cache_file):
        print(f">>> 发现缓存数据 {args.cache_file}，直接加载使用...")
        test_samples = np.load(args.cache_file)
    else:
        if args.root is None:
            print("错误: 缓存文件不存在，请使用 --root 指定原始数据路径来生成缓存。")
            return
        
        print(">>> 正在提取并保存 100 个测试样本到本地...")
        # 此处导入你刚刚修改的纯 NumPy 版加载器
        from stratified_dataset import StratifiedECGSegments
        ds_test = StratifiedECGSegments(root=args.root, split='test')
        
        # 提取前 100 个
        temp_data = []
        for i in range(min(100, len(ds_test))):
            x_raw, _ = ds_test[i]  # 得到 (1, 720) 的 NumPy 数组
            # 转换为 TFLite 默认习惯的 [1, 720, 1] 形状
            # (1, 720) -> (720, 1) -> (1, 720, 1)
            x_t = x_raw.transpose(1, 0)[np.newaxis, :].astype(np.float32)
            temp_data.append(x_t)
        
        # 将列表堆叠成一个 4D 数组: [100, 1, 720, 1]
        test_samples = np.array(temp_data)
        np.save(args.cache_file, test_samples)
        print(f">>> 100个样本已固化至: {args.cache_file}")

    # --- 第二部分：批量执行测试 ---
    target_models = ['liteecgnet', 'ldcnn', 'deepecgnet']
    all_results = []

    for m_name in target_models:
        tflite_file = f"{m_name}_sim_float32.tflite"
        tflite_path = os.path.join(args.model_dir, tflite_file)

        if not os.path.exists(tflite_path):
            print(f"[跳过] 找不到模型: {tflite_path}")
            continue

        print(f"\n>>> 运行模型: {m_name}")
        res_queue = Queue()
        # 传入加载的 100 个 NumPy 样本
        p = Process(target=worker_measure, args=(tflite_path, test_samples, m_name, 5, res_queue))
        p.start()
        
        res = res_queue.get()
        p.join()

        if res["status"] == "success":
            del res["status"]
            all_results.append(res)
            print(f"  结果: {res['Average Time (ms)']}ms | {res['Peak Memory (MB)']}MB")
        else:
            print(f"  [失败] {res['message']}")

    # --- 第三部分：结果保存 ---
    with open("integrated_performance_report.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print("\n>>> 整合评测报告已生成。")

if __name__ == "__main__":
    main()
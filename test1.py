import os
import glob
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 配置
# =========================
root_dir = "./data/mitbih"
output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

segment_sec = 2.0   # 每个样本显示 2 秒
lead_idx = 0        # 只画第1导联，更清晰；想画第2导联可改成 1

# AAMI 五分类顺序
class_order = ["N", "S", "V", "F", "Q"]

# MIT-BIH atr 原始符号 -> AAMI 五分类
AAMI_MAP = {
    # N
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # S
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # V
    "V": "V",
    "E": "V",

    # F
    "F": "F",

    # Q
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "?": "Q",
}

CLASS_COLOR = {
    "N": "green",
    "S": "orange",
    "V": "red",
    "F": "blue",
    "Q": "purple",
}

def get_record_list(root_dir):
    """获取所有 record 名称，比如 100, 101, 102..."""
    hea_files = glob.glob(os.path.join(root_dir, "*.hea"))
    record_names = [os.path.splitext(os.path.basename(p))[0] for p in hea_files]
    return sorted(record_names)

def symbol_to_aami(sym):
    """把 atr 原始符号映射为 AAMI 五分类"""
    return AAMI_MAP.get(sym, None)

def pick_one_example_per_class(root_dir, class_order):
    """
    遍历所有记录，为每个 AAMI 类别找第一个出现的样本
    返回：
        examples[cls] = {
            'record_name': ...,
            'sample': ...,
            'symbol': ...
        }
    """
    examples = {cls: None for cls in class_order}
    record_names = get_record_list(root_dir)

    print(f"Found {len(record_names)} records.")

    for record_name in record_names:
        record_path = os.path.join(root_dir, record_name)

        try:
            ann = wfdb.rdann(record_path, "atr")
        except Exception as e:
            print(f"[Skip] {record_name}: cannot read atr ({e})")
            continue

        for s, sym in zip(ann.sample, ann.symbol):
            cls = symbol_to_aami(sym)
            if cls in examples and examples[cls] is None:
                examples[cls] = {
                    "record_name": record_name,
                    "sample": int(s),
                    "symbol": sym
                }
                print(f"[Found] class={cls}, record={record_name}, sample={s}, symbol={sym}")

                if all(v is not None for v in examples.values()):
                    return examples

    return examples

def plot_example(ax, root_dir, example, cls_name, segment_sec=2.0, lead_idx=0):
    """
    以 example 的 sample 为中心，画 2 秒片段
    """
    record_name = example["record_name"]
    center_sample = example["sample"]
    center_symbol = example["symbol"]

    record_path = os.path.join(root_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    signal = record.p_signal
    fs = int(record.fs)
    sig_len, n_channels = signal.shape

    if lead_idx >= n_channels:
        lead_idx = 0

    half_len = int((segment_sec / 2) * fs)
    start_sample = max(center_sample - half_len, 0)
    end_sample = min(center_sample + half_len, sig_len)

    # 尽量补足到 2 秒
    target_len = int(segment_sec * fs)
    if end_sample - start_sample < target_len:
        if start_sample == 0:
            end_sample = min(target_len, sig_len)
        elif end_sample == sig_len:
            start_sample = max(sig_len - target_len, 0)

    seg = signal[start_sample:end_sample, lead_idx]
    seg_len = len(seg)
    t = np.arange(seg_len) / fs

    # 片段内的标注
    seg_ann_idx = [i for i, s in enumerate(ann.sample) if start_sample <= s < end_sample]
    seg_ann_samples = [ann.sample[i] for i in seg_ann_idx]
    seg_ann_symbols = [ann.symbol[i] for i in seg_ann_idx]

    # 画信号
    ax.plot(t, seg, color="black", linewidth=1.0)
    ax.grid(True, alpha=0.3)

    # 画标注
    y_min, y_max = float(np.min(seg)), float(np.max(seg))
    y_range = y_max - y_min + 1e-6
    ax.set_ylim(y_min - 0.25 * y_range, y_max + 0.45 * y_range)

    for s, sym in zip(seg_ann_samples, seg_ann_symbols):
        x = (s - start_sample) / fs
        cls = symbol_to_aami(sym)
        color = CLASS_COLOR.get(cls, "gray")

        ax.axvline(x=x, color=color, linestyle="--", alpha=0.7, linewidth=1.0)

        # 只把中心点标得更醒目
        if s == center_sample:
            y0 = seg[s - start_sample]
            ax.scatter([x], [y0], s=60, color="red", zorder=5)
            ax.text(
                x,
                y0 + 0.12 * y_range,
                f"{cls}\n{sym}",
                color="red",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold"
            )

    ax.set_title(
        f"AAMI Class: {cls_name} | Record(Patient): {record_name} | center symbol: {center_symbol}",
        fontsize=11
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Lead {lead_idx + 1}")

# =========================
# 主流程：找每个类别一个例子
# =========================
examples = pick_one_example_per_class(root_dir, class_order)

print("\nSelected examples:")
for cls in class_order:
    print(cls, "->", examples[cls])

# =========================
# 画成 2x3 总图
# =========================
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for i, cls in enumerate(class_order):
    ax = axes[i]
    if examples[cls] is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No example found for {cls}", ha="center", va="center", fontsize=12)
    else:
        plot_example(
            ax=ax,
            root_dir=root_dir,
            example=examples[cls],
            cls_name=cls,
            segment_sec=segment_sec,
            lead_idx=lead_idx
        )

# 多出来的最后一个子图关掉
for j in range(len(class_order), len(axes)):
    axes[j].axis("off")

plt.tight_layout()

save_path = os.path.join(output_dir, "mitbih_one_example_per_aami_class.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved to: {save_path}")
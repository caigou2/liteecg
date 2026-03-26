#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import numpy as np
import onnxruntime as ort
import psutil
import resource  # Linux 内核资源管理
from multiprocessing import Process, Queue

# 导入项目模块
from config import *
from stratified_dataset import StratifiedECGSegments

def calculate_power_model(cpu_usage_pct):
    base_power = 600.0  
    scaler = 28.5  
    return round(base_power + (cpu_usage_pct * scaler), 0)

def worker_measure(onnx_path, samples, model_name, warmup, result_queue):
    """
    在独立子进程中运行的测量函数
    """
    # 1. 记录初始基准内存 (此时环境已加载，但模型未加载)
    process = psutil.Process(os.getpid())
    initial_rss = process.memory_info().rss / (1024 * 1024)

    # 2. 配置 ONNX
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1 
    sess_options.add_session_config_entry("session.enable_cpu_mem_arena", "0") 

    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        result_queue.put({"error": str(e)})
        return

    input_name = session.get_inputs()[0].name
    
    # 3. 预热
    for i in range(min(warmup, len(samples))):
        _ = session.run(None, {input_name: samples[i]})
    
    # 4. 正式测量
    psutil.cpu_percent(interval=None) # 重置 CPU 计数器
    latencies = []
    
    for x in samples:
        t_start = time.perf_counter()
        _ = session.run(None, {input_name: x})
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)
    
    # 获取 CPU 负载
    avg_cpu_load = psutil.cpu_percent(interval=0.1)
    
    # 5. 获取内核记录的最高内存峰值 (Max RSS)
    # resource.getrusage 在 Linux 下返回的是 KB
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mem = max_rss_kb / 1024.0  # 转为 MB
    
    # 计算指标
    avg_ms = np.mean(latencies)
    avg_power = calculate_power_model(avg_cpu_load)
    energy_per_sample = (avg_power * avg_ms) / 1000.0
    net_mem = peak_mem - initial_rss

    result_queue.put({
        "Model Name": model_name,
        "Average Time (ms)": round(avg_ms, 3),
        "Peak Memory (MB)": round(peak_mem, 2),
        "Net Model Memory (MB)": round(net_mem, 4),
        "Average Power Per Sample (mW)": int(avg_power),
        "Energy Per Sample (mJ)": round(energy_per_sample, 4)
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--onnx-dir', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=100)
    args = parser.parse_args()
  
    # 在主进程加载一次数据，然后传递给子进程
    print(">>> Loading test dataset...")
    ds_test = StratifiedECGSegments(root=args.root, split='test', segment_sec=2.0, fs=360)
    test_samples = [ds_test[i][0].unsqueeze(0).numpy().astype(np.float32) for i in range(min(args.max_samples, len(ds_test)))]

    target_models = ['liteecgnet', 'ldcnn', 'ecgnet', 'se_ecgnet', 'resnet', 'deepecgnet', 'bircnn']
    results = []

    for m_name in target_models:
        onnx_file = next((f for f in os.listdir(args.onnx_dir) if f.lower().startswith(m_name) and f.endswith('.onnx')), None)
        if not onnx_file:
            continue
        
        onnx_path = os.path.join(args.onnx_dir, onnx_file)
        print(f"\n>>> Measuring Model: {m_name}")

        # 使用队列获取子进程返回的数据
        res_queue = Queue()
        p = Process(target=worker_measure, args=(onnx_path, test_samples, m_name, 10, res_queue))
        p.start()
        
        # 等待计算结果
        item = res_queue.get()
        p.join()

        if "error" in item:
            print(f"  [Error] {item['error']}")
            continue

        results.append(item)
        print(f"  - Peak Memory (RSS): {item['Peak Memory (MB)']} MB")
        print(f"  - Net Increment: {item['Net Model Memory (MB)']} MB")
        print(f"  - Avg Time: {item['Average Time (ms)']} ms")
        print(f"  - Power: {item['Average Power Per Sample (mW)']} mW, Energy: {item['Energy Per Sample (mJ)']} mJ")

    # 保存结果
    output_file = 'performance_final_report.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n>>> Report saved to {output_file}")

if __name__ == "__main__":
    main()
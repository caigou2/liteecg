import onnxruntime as ort
import numpy as np
import time

def benchmark_onnx(onnx_path, num_tests=100):
    # 1. 创建推理会话 (Session)
    # 对于 CPU，ONNX Runtime 会自动调用多线程和算子融合
    session = ort.InferenceSession(onnx_path)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # 2. 准备测试数据 (注意输入类型必须是 float32)
    data = np.random.randn(*input_shape).astype(np.float32)

    # 3. 预热 (Warmup)
    for _ in range(10):
        _ = session.run(None, {input_name: data})

    # 4. 正式测量
    start_time = time.perf_counter()
    for _ in range(num_tests):
        _ = session.run(None, {input_name: data})
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) * 1000.0) / num_tests
    return avg_ms

# 执行对比
lite_ms = benchmark_onnx('./weights/liteecgnet.onnx')
ldcnn_ms = benchmark_onnx('./weights/ldcnn.onnx')

print(f"ONNX LiteECGNet Latency: {lite_ms:.4f} ms")
print(f"ONNX LDCNN Latency: {ldcnn_ms:.4f} ms")
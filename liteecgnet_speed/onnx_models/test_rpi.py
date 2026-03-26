import numpy as np
import tflite_runtime.interpreter as tflite
import time

# 1. 加载模型
model_path = "liteecgnet1_sim_float32.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 2. 获取输入和输出张量详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. 准备一批随机数据进行测试 (B=1, C=1, T=720)
input_shape = input_details[0]['shape']
input_data = np.random.standard_normal(input_shape).astype(np.float32)

# 4. 执行推理并计时
interpreter.set_tensor(input_details[0]['index'], input_data)
start_time = time.time()
interpreter.invoke()
end_time = time.time()

# 5. 获取结果
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"模型推理成功！")
print(f"输入形状: {input_shape}")
print(f"预测分类结果: {output_data}")
print(f"单次推理耗时: {(end_time - start_time) * 1000:.2f} ms")
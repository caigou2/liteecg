# 1. 基础镜像
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. 设置工作目录
WORKDIR /app

# 3. 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4. 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 复制项目代码
COPY . .

# 6. 运行脚本
CMD ["python", "main.py"]

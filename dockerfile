# 第一阶段：构建环境
FROM python:3.10-slim as builder

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装Python依赖
COPY requirements.txt /tmp/
RUN pip install --user --no-warn-script-location -r /tmp/requirements.txt

# 第二阶段：运行环境
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# 安装运行时依赖（最小化）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制Python包
COPY --from=builder /root/.local /root/.local

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "cloudpose_server:app", "--host", "0.0.0.0", "--port", "8000"]
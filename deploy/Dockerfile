# ============================================
# Dockerfile - 智能客服系统容器化部署
# 
# 构建命令: docker build -t customer-service-bot .
# 运行命令: docker run -p 8000:8000 -e OPENAI_API_KEY=your-key customer-service-bot
# ============================================

# 使用Python 3.11官方镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
# gcc和build-essential用于编译Python包
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY data/ ./data/
COPY main.py .

# 创建向量数据库目录
RUN mkdir -p ./data/vector_db

# 暴露端口（用于API服务）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('healthy')" || exit 1

# 默认启动 API 服务
CMD ["python", "main.py", "--api", "--host", "0.0.0.0", "--port", "8000"]

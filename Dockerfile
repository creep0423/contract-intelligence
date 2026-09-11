# syntax=docker/dockerfile:1.7

# 基础镜像：基于 Python 3.12 的 Contract Intelligence 运行时镜像
ARG APP_BASE_IMAGE=contract-intelligence-base:py312
# PyPI 镜像源地址（可通过构建参数覆盖，用于离线环境或国内加速）
ARG PIP_INDEX_URL=https://pypi.org/simple
# PyTorch CPU 版安装源（GPU 版可替换为 cu118/cu121 索引）
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
FROM ${APP_BASE_IMAGE}

LABEL org.opencontainers.image.title="Contract Intelligence" \
      org.opencontainers.image.description="Enterprise contract performance and risk intelligence" \
      org.opencontainers.image.source="contract-intelligence"

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# 禁止 Python 生成 .pyc 字节码文件，减少镜像层体积和容器启动时的写操作
ENV PYTHONDONTWRITEBYTECODE=1
# 禁用 Python 输出缓冲，确保 uvicorn 日志实时输出到 Docker 日志流
ENV PYTHONUNBUFFERED=1
# 强制 Python 使用 UTF-8 编码，避免中文字符在容器内产生编码异常
ENV PYTHONIOENCODING=utf-8
# 禁用 HuggingFace Tokenizers 的多进程并行，避免在 gunicorn/uvicorn 多 worker 模式下产生死锁
ENV TOKENIZERS_PARALLELISM=false
# 关闭 HuggingFace Hub 遥测上报，生产环境无需发送用量统计
ENV HF_HUB_DISABLE_TELEMETRY=1
# 让 GitPython 在非 git 目录下不输出警告日志
ENV GIT_PYTHON_REFRESH=quiet

# 设置容器内工作目录
WORKDIR /app

# 先复制 requirements.txt，利用 Docker 分层缓存：依赖未变时不重新安装
COPY requirements.txt ./
# 挂载 pip 缓存目录加速构建；安装后执行 pip check 验证依赖无冲突；
# 最后导入所有核心包确保启动时不会因缺失依赖而崩溃
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url "${PIP_INDEX_URL}" --extra-index-url "${PYTORCH_INDEX_URL}" -r requirements.txt \
    && python -m pip check \
    && python -c "import fastapi, uvicorn, langchain, langchain_milvus, pymilvus, sentence_transformers, transformers, safetensors, torch, pandas, fitz, docx, pptx, docling, ragas, markdown, mkdocs, redis; print('runtime dependencies ok')"

# 复制项目代码到容器（.dockerignore 控制排除项）
COPY . .

# 声明容器监听端口 8000（FastAPI 默认端口）
EXPOSE 8000

# 容器启动命令：使用 uvicorn 运行 FastAPI 应用，监听所有网络接口
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

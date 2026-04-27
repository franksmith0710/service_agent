#!/bin/bash
set -e

echo "========================================="
echo "  智能客服 Agent 启动脚本"
echo "========================================="

# 配置
POSTGRES_HOST="${POSTGRES_HOST:-db}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# 1. 等待 PostgreSQL 就绪
echo "[1/4] 等待 PostgreSQL 就绪..."
retries=0
until PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-kefu_agent" -c '\q' 2>/dev/null; do
    retries=$((retries + 1))
    if [ $retries -ge $MAX_RETRIES ]; then
        echo "错误: PostgreSQL 连接超时"
        exit 1
    fi
    echo "  等待中... ($retries/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done
echo "  PostgreSQL 已就绪"

# 2. 等待 Ollama 就绪
echo "[2/4] 等待 Ollama 就绪..."
retries=0
until curl -s "$OLLAMA_BASE_URL/api/version" > /dev/null 2>&1; do
    retries=$((retries + 1))
    if [ $retries -ge $MAX_RETRIES ]; then
        echo "错误: Ollama 连接超时"
        exit 1
    fi
    echo "  等待中... ($retries/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done
echo "  Ollama 已就绪"

# 3. 检查并初始化知识库
echo "[3/4] 检查知识库..."
cd /app
python3 -c "
import sys
sys.path.insert(0, '/app')
from src.services.rag import get_rag, init_from_files

rag = get_rag()
docs = rag.similarity_search('测试', k=1)
if not docs:
    print('  知识库为空，正在初始化...')
    init_from_files()
    print('  知识库初始化完成')
else:
    print('  知识库已存在')
"
echo "  知识库检查完成"

# 4. 启动服务
echo "[4/4] 启动服务..."
echo "  - FastAPI: http://localhost:8000"
echo "  - Streamlit: http://localhost:8501"

# 使用 trap 优雅关闭
trap 'echo "收到关闭信号，正在停止服务..."; kill $API_PID $STREAMLIT_PID; wait; exit 0' SIGINT SIGTERM

# 后台启动 FastAPI
cd /app
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# 后台启动 Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
STREAMLIT_PID=$!

# 等待两个进程
wait $API_PID $STREAMLIT_PID
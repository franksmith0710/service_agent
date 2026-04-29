#!/bin/bash
set -e

echo "========================================="
echo "  智能客服 Agent 启动脚本"
echo "========================================="

# 配置
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"

# 1. 检查/拉取嵌入模型
echo "[1/3] 检查嵌入模型..."
if ! curl -s "$OLLAMA_BASE_URL/api/tags" | grep -q "\"name\":\"$EMBEDDING_MODEL\""; then
    echo "  正在拉取嵌入模型: $EMBEDDING_MODEL ..."
    curl -X POST "$OLLAMA_BASE_URL/api/pull" -d "{\"name\":\"$EMBEDDING_MODEL\"}" --no-buffer
    echo "  嵌入模型拉取完成"
else
    echo "  嵌入模型已存在: $EMBEDDING_MODEL"
fi

# 2. 检查/初始化知识库
echo "[2/3] 检查知识库..."
cd /app
python -c "
import sys
sys.path.insert(0, '/app')
from src.services.rag import get_rag, init_from_files

try:
    rag = get_rag()
    docs = rag.similarity_search('测试', k=1)
    if not docs:
        print('  知识库为空，正在初始化...')
        count = init_from_files()
        print(f'  知识库初始化完成，导入 {count} 个文档')
    else:
        print('  知识库已存在')
except Exception as e:
    print(f'  知识库检查失败: {e}')
    sys.exit(1)
"
echo "  知识库检查完成"

# 3. 启动服务
echo "[3/3] 启动服务..."
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

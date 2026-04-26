"""
FastAPI 应用入口
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import config
from src.config.logger import setup_logger
from src.services.agent import run_agent
from src.services.memory import get_memory, clear_memory

# 初始化日志
logger = setup_logger(name="kefu_agent", level=logging.INFO)

# 创建 FastAPI 应用
app = FastAPI(
    title="智能客服 API",
    description="基于 LangChain 的 AI 智能客服聊天机器人 API",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """聊天请求"""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""

    session_id: str
    response: str
    success: bool = True
    error: Optional[str] = None


class ClearRequest(BaseModel):
    """清空对话请求"""

    session_id: str


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "智能客服 API",
        "version": "1.0.0",
        "dispatch_model": "deepseek-r1:1.5b",
        "generation_provider": "siliconflow",
        "generation_model": config.siliconflow.model,
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    Args:
        request: 聊天请求

    Returns:
        聊天响应
    """
    try:
        logger.info(
            f"Chat request: session={request.session_id}, message={request.message[:50]}..."
        )

        # 运行 Agent
        full_response = ""
        for chunk in run_agent(request.session_id, request.message):
            full_response += chunk

        logger.info(f"Chat response completed for session {request.session_id}")

        return ChatResponse(
            session_id=request.session_id,
            response=full_response,
            success=True,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            session_id=request.session_id,
            response="",
            success=False,
            error=str(e),
        )


@app.post("/clear")
async def clear(request: ClearRequest):
    """
    清空对话历史

    Args:
        request: 清空请求

    Returns:
        结果
    """
    try:
        clear_memory(request.session_id)
        logger.info(f"Cleared memory for session {request.session_id}")
        return {"success": True, "message": f"Session {request.session_id} cleared"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    获取对话历史

    Args:
        session_id: 会话 ID

    Returns:
        对话历史
    """
    try:
        memory = get_memory(session_id)
        messages = memory.get_messages()

        history = []
        for msg in messages:
            history.append(
                {
                    "type": "user" if msg.type == "human" else "assistant",
                    "content": msg.content,
                }
            )

        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ============================================
# FastAPI 路由模块
# 提供 RESTful API 供前端调用
# ============================================

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config
from chatbot import CustomerServiceBot

logger = logging.getLogger(__name__)

# 全局 bot 实例（模块加载后由 create_app 初始化）
bot: Optional[CustomerServiceBot] = None


# ============================================
# 请求/响应模型
# ============================================
class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str = Field(
        ...,
        description="会话ID，通过 /session/new 获取",
        min_length=1,
        max_length=100
    )
    message: str = Field(
        ...,
        description="用户消息",
        min_length=1,
        max_length=2000
    )


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(..., description="助手回答")
    intent: str = Field(..., description="意图类别")
    needs_human: bool = Field(..., description="是否需要转人工")
    session_id: str = Field(..., description="会话ID")
    turn_count: int = Field(..., description="当前轮数")


class SessionResponse(BaseModel):
    """新建会话响应"""
    session_id: str = Field(..., description="新会话ID")


class SessionInfoResponse(BaseModel):
    """会话信息响应"""
    session_id: str
    is_valid: bool
    message_count: int
    turn_count: int
    last_active: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str


# ============================================
# 应用工厂
# ============================================
def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    global bot

    app = FastAPI(
        title="LangChain 智能客服系统 API",
        description=(
            "企业级智能客服系统，支持 RAG 检索增强生成、"
            "意图识别、多轮对话管理"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================
    # 生命周期事件
    # ==========================================
    @app.on_event("startup")
    async def startup():
        """应用启动：初始化客服机器人"""
        global bot
        validation = Config.validate()
        if not validation["valid"]:
            error_msg = "; ".join(validation["errors"])
            raise RuntimeError(f"配置验证失败：{error_msg}")
        for warning in validation["warnings"]:
            logger.warning(f"⚠ 配置警告：{warning}")

        bot = CustomerServiceBot()
        logger.info("API 服务启动完成")

    @app.on_event("shutdown")
    async def shutdown():
        """应用关闭"""
        logger.info("API 服务正在关闭...")

    # ==========================================
    # 路由
    # ==========================================
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["系统"],
        summary="健康检查"
    )
    async def health():
        """检查服务是否正常运行"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )

    @app.post(
        "/session/new",
        response_model=SessionResponse,
        tags=["会话"],
        summary="创建新会话"
    )
    async def new_session():
        """创建一个新的对话会话，返回 session_id"""
        if not bot:
            raise HTTPException(status_code=503, detail="服务未就绪，请稍后重试")

        session_id = bot.create_session()
        return SessionResponse(session_id=session_id)

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["对话"],
        summary="发送对话消息"
    )
    async def chat(request: ChatRequest):
        """
        发送用户消息并获取 AI 回复

        - **session_id**: 通过 `/session/new` 获取的会话ID
        - **message**: 用户输入的问题或消息
        """
        if not bot:
            raise HTTPException(status_code=503, detail="服务未就绪，请稍后重试")

        try:
            result = bot.chat(request.session_id, request.message)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"对话处理失败：{e}")
            raise HTTPException(status_code=500, detail="内部处理错误，请稍后重试")

        return ChatResponse(**result)

    @app.get(
        "/session/{session_id}/info",
        response_model=SessionInfoResponse,
        tags=["会话"],
        summary="获取会话信息"
    )
    async def session_info(session_id: str):
        """查看指定会话的详细信息"""
        if not bot:
            raise HTTPException(status_code=503, detail="服务未就绪")

        info = bot.get_session_info(session_id)
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])

        return SessionInfoResponse(**info)

    return app

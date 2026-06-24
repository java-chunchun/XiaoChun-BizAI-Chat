# ============================================
# 配置文件
# 包含系统配置、API密钥、模型参数等
# ============================================

import os
from pathlib import Path
from typing import Dict, Any

# 自动加载项目根目录的 .env 文件
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

class Config:
    """
    系统配置类
    集中管理所有配置项，便于维护和修改
    """
    
    # ============================================
    # LLM API 配置（兼容 OpenAI / DeepSeek / 智谱 等）
    # ============================================
    # API 密钥 — 设置环境变量或直接填写
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

    # API 基础地址 — 换成国内厂商的地址即可切换模型
    # DeepSeek:  https://api.deepseek.com/v1       (推荐，便宜好用)
    # 智谱:      https://open.bigmodel.cn/api/paas/v4
    # OpenAI:    https://api.openai.com/v1          (需要 VPN)
    OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.deepseek.com/v1"  # 默认使用 DeepSeek，国内直连
    )

    # ============================================
    # 模型配置
    # ============================================
    # 主模型：用于生成回答
    # DeepSeek: deepseek-v4-flash (快) / deepseek-v4-pro (强)
    # 智谱: glm-4-flash   OpenAI: gpt-4-turbo
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "deepseek-chat")

    # 备用模型：当主模型不可用时使用（意图分类也用这个）
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek-chat")

    # 温度参数：0-2之间，越低越稳定，越高越有创意
    TEMPERATURE = 0.3
    # 最大 token 数
    MAX_TOKENS = 2000

    # ============================================
    # Embedding / 向量化配置
    # ============================================
    # 向量化方案： "local"（免费本地模型）或 "openai"（需 API）
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

    # 本地 Embedding 模型（仅 EMBEDDING_PROVIDER="local" 时生效）
    # bge-small-zh-v1.5: 中文优化，95MB，CPU 即可跑
    # bge-large-zh-v1.5: 精度更高，但 1.3GB
    LOCAL_EMBEDDING_MODEL = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5"
    )

    # OpenAI 兼容的 Embedding 模型名（EMBEDDING_PROVIDER="openai" 时使用）
    # OpenAI: text-embedding-ada-002   智谱: embedding-2
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # ============================================
    # 向量数据库配置
    # ============================================
    # 向量数据库持久化路径
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    # 检索时返回的文档数量
    RETRIEVAL_TOP_K = 3
    # 相似度阈值，低于此值的文档将被过滤
    SIMILARITY_THRESHOLD = 0.7
    
    # ============================================
    # 系统行为配置
    # ============================================
    # 是否启用调试模式
    DEBUG = False
    # 日志级别：DEBUG, INFO, WARNING, ERROR
    LOG_LEVEL = "INFO"
    # 最大对话轮数
    MAX_CONVERSATION_TURNS = 10
    # 会话超时时间（秒）
    SESSION_TIMEOUT = 1800
    
    # ============================================
    # 客服系统配置
    # ============================================
    # 转人工阈值：当置信度低于此值时建议转人工
    HUMAN_HANDOFF_THRESHOLD = 0.6
    # 欢迎语
    WELCOME_MESSAGE = "您好！我是智能客服助手，请问有什么可以帮您？"
    # 无法回答时的回复模板
    UNKNOWN_ANSWER_TEMPLATE = (
        "抱歉，我暂时无法回答这个问题。"
        "您可以：\n"
        "1. 换个方式描述您的问题\n"
        "2. 联系人工客服：400-888-9999\n"
        "3. 发送邮件至：support@company.com"
    )
    
    # ============================================
    # 数据文件路径
    # ============================================
    KNOWLEDGE_BASE_FILE = "./data/knowledge_base.txt"
    PRODUCTS_FILE = "./data/products.json"
    # FAQ数据已整合在 products.json 中，不需要独立文件
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        便于序列化和日志记录
        """
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """
        验证配置完整性

        返回：
            {"valid": bool, "errors": list, "warnings": list}
        """
        errors = []
        warnings = []

        # 1. 检查API密钥
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your-api-key-here":
            errors.append(
                "OPENAI_API_KEY 未设置。"
                "国内用户推荐注册 DeepSeek（platform.deepseek.com，新用户送 500 万 token）：\n"
                "  export OPENAI_API_KEY='sk-你的密钥'\n"
                "  export OPENAI_BASE_URL='https://api.deepseek.com/v1'"
            )

        # 2. 检查数据文件是否存在
        if not os.path.exists(cls.KNOWLEDGE_BASE_FILE):
            warnings.append(f"知识库文件不存在：{cls.KNOWLEDGE_BASE_FILE}")
        if not os.path.exists(cls.PRODUCTS_FILE):
            warnings.append(f"产品文件不存在：{cls.PRODUCTS_FILE}")

        # 3. 检查本地 Embedding 依赖（如果使用 local 模式）
        if cls.EMBEDDING_PROVIDER == "local":
            try:
                import langchain_huggingface  # noqa: F401
            except ImportError:
                warnings.append(
                    "使用本地 Embedding 需要安装依赖："
                    "pip install langchain-huggingface sentence-transformers"
                )

        # 4. 检查数值范围
        if not (0 <= cls.SIMILARITY_THRESHOLD <= 1):
            errors.append(
                f"SIMILARITY_THRESHOLD 应在 0-1 之间，当前值：{cls.SIMILARITY_THRESHOLD}"
            )
        if not (0 <= cls.TEMPERATURE <= 2):
            errors.append(
                f"TEMPERATURE 应在 0-2 之间，当前值：{cls.TEMPERATURE}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# ============================================
# 提示词模板配置
# ============================================
class PromptTemplates:
    """
    提示词模板配置
    统一管理所有提示词，便于维护和优化
    """
    
    # 系统提示词：定义AI助手的角色和行为
    SYSTEM_PROMPT = """你是一位专业、友好的企业客服助手。你的职责是：
1. 准确回答用户关于产品、服务、价格的问题
2. 使用礼貌、亲切的语言风格
3. 如果无法回答问题，诚实地告知用户并提供替代方案
4. 不要编造信息，只基于提供的知识库回答
5. 回答要简洁明了，避免冗长

当前日期：{current_date}
"""
    
    # RAG检索提示词：用于增强检索后的回答生成
    RAG_PROMPT = """基于以下参考信息，回答用户的问题。

参考信息：
{context}

用户问题：{question}

要求：
1. 如果参考信息足够，直接基于信息回答
2. 如果信息不足，明确告知用户
3. 保持友好、专业的语气
4. 回答控制在200字以内

回答："""
    
    # 意图识别提示词：判断用户意图
    INTENT_PROMPT = """分析用户的输入，判断其意图类别。

用户输入：{input}

可能的意图类别：
- product_inquiry: 产品咨询
- price_inquiry: 价格咨询
- technical_support: 技术支持
- complaint: 投诉反馈
- general_chat: 闲聊
- human_handoff: 要求转人工

请只输出意图类别，不要解释。"""
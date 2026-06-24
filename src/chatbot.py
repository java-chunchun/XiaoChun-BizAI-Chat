# ============================================
# 智能客服核心模块
# 实现对话管理、意图识别、回答生成等功能
# ============================================

import json
import logging
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI

# 兼容不同版本的 langchain 导入路径
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage

from config import Config, PromptTemplates
from knowledge_base import KnowledgeBase

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# 工具函数
# ============================================
def sanitize_input(text: str, max_length: int = 2000) -> str:
    """
    过滤用户输入，防止注入和过长的恶意输入

    参数：
        text: 原始输入文本
        max_length: 最大允许长度

    返回：
        过滤并截断后的安全文本
    """
    if not text or not text.strip():
        return ""

    # 截断超长输入
    if len(text) > max_length:
        text = text[:max_length]

    # 移除控制字符（保留换行、制表符和常用标点）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


class ConversationManager:
    """
    对话管理器
    
    功能：
    1. 维护多轮对话历史
    2. 管理会话状态（创建、更新、过期检查）
    3. 控制对话轮数，防止无限对话
    
    使用示例：
        manager = ConversationManager()
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "你好")
        history = manager.get_history(session_id)
    """
    
    def __init__(self):
        """
        初始化对话管理器
        创建空的会话存储字典
        """
        # 会话存储：{session_id: {messages: [], last_active: timestamp, turn_count: 0}}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("对话管理器初始化完成")
    
    def create_session(self) -> str:
        """
        创建新的对话会话
        
        返回：
            新生成的会话ID（基于时间戳）
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.sessions[session_id] = {
            "messages": [],           # 消息历史列表
            "last_active": datetime.now(),  # 最后活跃时间
            "turn_count": 0           # 当前对话轮数
        }
        logger.info(f"创建新会话：{session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        添加消息到指定会话
        
        参数：
            session_id: 会话ID
            role: 消息角色（"user"或"assistant"）
            content: 消息内容
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在：{session_id}")
        
        # 添加消息到历史
        self.sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 更新活跃时间和轮数
        self.sessions[session_id]["last_active"] = datetime.now()
        if role == "user":
            self.sessions[session_id]["turn_count"] += 1
        
        logger.debug(f"会话{session_id}添加消息：{role} - {content[:50]}...")
    
    def get_history(self, session_id: str, max_turns: int = None) -> List[Dict[str, str]]:
        """
        获取指定会话的历史消息
        
        参数：
            session_id: 会话ID
            max_turns: 最大返回轮数，None表示返回全部
        
        返回：
            消息列表，格式：[{"role": "user", "content": "..."}, ...]
        """
        if session_id not in self.sessions:
            return []
        
        messages = self.sessions[session_id]["messages"]
        
        # 如果超过最大轮数，只保留最近的
        if max_turns and len(messages) > max_turns * 2:
            messages = messages[-max_turns * 2:]
        
        return [{"role": m["role"], "content": m["content"]} for m in messages]
    
    def is_session_valid(self, session_id: str) -> bool:
        """
        检查会话是否有效（未过期）
        
        参数：
            session_id: 会话ID
        
        返回：
            True表示有效，False表示已过期
        """
        if session_id not in self.sessions:
            return False
        
        last_active = self.sessions[session_id]["last_active"]
        elapsed = (datetime.now() - last_active).total_seconds()
        
        return elapsed < Config.SESSION_TIMEOUT
    
    def cleanup_expired_sessions(self) -> int:
        """
        清理过期的会话
        
        返回：
            清理的会话数量
        """
        expired = [
            sid for sid in self.sessions
            if not self.is_session_valid(sid)
        ]
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            logger.info(f"清理了{len(expired)}个过期会话")
        return len(expired)


class IntentClassifier:
    """
    意图分类器
    
    使用大模型判断用户输入的意图类别
    支持：产品咨询、价格咨询、技术支持、投诉、闲聊、转人工
    
    使用示例：
        classifier = IntentClassifier()
        intent = classifier.classify("你们的产品多少钱？")
        # 返回："price_inquiry"
    """
    
    def __init__(self):
        """
        初始化意图分类器
        使用轻量级模型进行分类，降低成本
        """
        self.llm = ChatOpenAI(
            model=Config.FALLBACK_MODEL,  # 使用gpt-3.5-turbo，更快更便宜
            temperature=0,                 # 温度设为0，保证结果稳定
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            request_timeout=15             # 15秒超时
        )
        logger.info("意图分类器初始化完成")

    def _invoke_with_retry(self, messages: list, max_retries: int = 2, backoff: float = 2.0):
        """
        带重试机制的LLM调用

        参数：
            messages: 消息列表
            max_retries: 最大重试次数
            backoff: 退避基数（秒）

        返回：
            LLM响应对象

        异常：
            RuntimeError: 所有重试均失败
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = backoff ** (attempt + 1)
                    logger.warning(
                        f"意图分类LLM调用失败（第{attempt+1}/{max_retries+1}次），"
                        f"{wait_time:.1f}秒后重试：{e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"意图分类LLM调用在{max_retries+1}次尝试后仍然失败")

        raise RuntimeError(f"意图分类LLM调用失败：{last_error}")

    def classify(self, user_input: str) -> str:
        """
        对用户输入进行意图分类

        参数：
            user_input: 用户的原始输入

        返回：
            意图类别字符串
        """
        # 构建分类提示词（使用正确的消息格式）
        prompt = PromptTemplates.INTENT_PROMPT.format(input=user_input)
        messages = [
            SystemMessage(content="你是一个意图分类助手。只输出意图类别名称，不要解释。"),
            HumanMessage(content=prompt)
        ]

        # 调用模型进行分类（带重试）
        try:
            response = self._invoke_with_retry(messages)
            intent = response.content.strip().lower()
        except RuntimeError as e:
            logger.error(f"意图分类失败，回退到 general_chat：{e}")
            return "general_chat"

        # 标准化意图名称
        valid_intents = [
            "product_inquiry", "price_inquiry", "technical_support",
            "complaint", "general_chat", "human_handoff"
        ]

        # 如果返回的意图不在列表中，默认为general_chat
        if intent not in valid_intents:
            intent = "general_chat"

        logger.info(f"意图识别：'{user_input[:30]}...' -> {intent}")
        return intent


class CustomerServiceBot:
    """
    智能客服机器人
    
    核心功能：
    1. 接收用户输入
    2. 识别用户意图
    3. 检索相关知识
    4. 生成回答
    5. 管理对话上下文
    
    使用示例：
        bot = CustomerServiceBot()
        session_id = bot.create_session()
        response = bot.chat(session_id, "智能客服系统多少钱？")
    """
    
    def __init__(self):
        """
        初始化客服机器人
        创建所有需要的组件
        """
        # 0. 验证配置
        validation = Config.validate()
        if not validation["valid"]:
            error_msg = "; ".join(validation["errors"])
            raise RuntimeError(f"配置验证失败：{error_msg}")
        for warning in validation["warnings"]:
            logger.warning(f"⚠ 配置警告：{warning}")

        # 1. 初始化大语言模型
        # 使用gpt-4-turbo作为主模型，temperature=0.3保证稳定性
        self.llm = ChatOpenAI(
            model=Config.PRIMARY_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            request_timeout=60              # 60秒超时
        )

        # 2. 初始化知识库
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.initialize()

        # 3. 初始化对话管理器
        self.conversation_manager = ConversationManager()

        # 4. 初始化意图分类器
        self.intent_classifier = IntentClassifier()

        logger.info("智能客服机器人初始化完成")

    def _invoke_with_retry(self, messages: list, max_retries: int = 2, backoff: float = 2.0):
        """
        带重试机制的LLM调用

        参数：
            messages: 消息列表
            max_retries: 最大重试次数
            backoff: 退避基数（秒）

        返回：
            LLM响应对象

        异常：
            RuntimeError: 所有重试均失败
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = backoff ** (attempt + 1)
                    logger.warning(
                        f"LLM调用失败（第{attempt+1}/{max_retries+1}次），"
                        f"{wait_time:.1f}秒后重试：{e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM调用在{max_retries+1}次尝试后仍然失败")

        raise RuntimeError(f"LLM调用失败：{last_error}")
    
    def create_session(self) -> str:
        """
        创建新的对话会话
        
        返回：
            新会话的ID
        """
        return self.conversation_manager.create_session()
    
    def chat(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        处理用户输入并生成回答
        
        这是核心对话方法，完整流程：
        1. 检查会话有效性
        2. 保存用户消息
        3. 识别用户意图
        4. 检索相关知识
        5. 生成回答
        6. 保存助手回复
        7. 返回结果
        
        参数：
            session_id: 会话ID
            user_input: 用户输入的文本
        
        返回：
            包含回答、意图、是否需要转人工等信息的字典
        """
        # 0. 过滤用户输入
        user_input = sanitize_input(user_input)
        if not user_input:
            return {
                "answer": "请输入您的问题，我很乐意帮助您。",
                "intent": "empty_input",
                "needs_human": False
            }

        # 1. 检查会话是否有效
        if not self.conversation_manager.is_session_valid(session_id):
            return {
                "answer": "会话已过期，请重新开始对话。",
                "intent": "session_expired",
                "needs_human": False
            }

        # 2. 保存用户消息到历史
        self.conversation_manager.add_message(session_id, "user", user_input)

        # 3. 识别用户意图
        intent = self.intent_classifier.classify(user_input)

        # 4. 根据意图决定处理方式
        try:
            if intent == "human_handoff":
                # 用户明确要求转人工
                answer = Config.UNKNOWN_ANSWER_TEMPLATE
                needs_human = True
            elif intent == "general_chat":
                # 闲聊模式，直接回答
                answer = self._generate_chat_response(session_id, user_input)
                needs_human = False
            else:
                # 业务咨询，使用RAG检索增强
                answer, needs_human = self._generate_rag_response(session_id, user_input)
        except RuntimeError as e:
            logger.error(f"回答生成失败：{e}")
            answer = "抱歉，系统暂时遇到问题，请稍后重试或联系人工客服。"
            needs_human = True

        # 5. 保存助手回复
        self.conversation_manager.add_message(session_id, "assistant", answer)

        # 6. 构建返回结果
        result = {
            "answer": answer,
            "intent": intent,
            "needs_human": needs_human,
            "session_id": session_id,
            "turn_count": self.conversation_manager.sessions[session_id]["turn_count"]
        }

        logger.info(f"对话完成：session={session_id}, intent={intent}, needs_human={needs_human}")
        return result
    
    def _generate_rag_response(self, session_id: str, user_input: str) -> tuple:
        """
        使用RAG（检索增强生成）生成回答
        
        流程：
        1. 从知识库检索相关文档
        2. 构建增强提示词（用户问题 + 检索结果）
        3. 调用大模型生成回答
        4. 判断是否需要转人工
        
        参数：
            session_id: 会话ID
            user_input: 用户问题
        
        返回：
            (回答文本, 是否需要转人工)
        """
        # 1. 检索相关知识
        search_results = self.knowledge_base.search(user_input)
        
        # 2. 判断检索结果是否足够
        if not search_results:
            # 没有找到相关知识，建议转人工
            return Config.UNKNOWN_ANSWER_TEMPLATE, True
        
        # 3. 构建上下文
        # 将检索结果拼接成文本
        context = "\n\n".join([
            f"[相关度{r['score']:.2f}] {r['content']}"
            for r in search_results
        ])
        
        # 4. 获取对话历史
        history = self.conversation_manager.get_history(session_id, max_turns=3)
        history_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
            for m in history[:-1]  # 排除当前问题
        ])
        
        # 5. 构建提示词
        prompt = PromptTemplates.RAG_PROMPT.format(
            context=context,
            question=user_input
        )
        
        # 如果有历史对话，添加进去
        if history_text:
            prompt = f"历史对话：\n{history_text}\n\n{prompt}"
        
        # 6. 调用模型生成回答
        messages = [
            SystemMessage(content=PromptTemplates.SYSTEM_PROMPT.format(current_date=datetime.now().date())),
            HumanMessage(content=prompt)
        ]
        
        response = self._invoke_with_retry(messages)
        answer = response.content.strip()
        
        # 7. 判断是否需要转人工
        # 如果回答中包含"无法回答"、"不知道"等，或者检索结果相似度都很低
        needs_human = (
            "无法" in answer or 
            "不知道" in answer or
            all(r["score"] < Config.HUMAN_HANDOFF_THRESHOLD for r in search_results)
        )
        
        return answer, needs_human
    
    def _generate_chat_response(self, session_id: str, user_input: str) -> str:
        """
        生成闲聊回复

        不检索知识库，直接基于对话历史生成回复
        适合问候、感谢、闲聊等场景

        参数：
            session_id: 会话ID
            user_input: 用户输入

        返回：
            生成的回复文本
        """
        # 获取对话历史（限制轮数，防止上下文膨胀）
        history = self.conversation_manager.get_history(
            session_id, max_turns=Config.MAX_CONVERSATION_TURNS
        )
        
        # 构建消息列表
        messages = [
            SystemMessage(content=PromptTemplates.SYSTEM_PROMPT.format(current_date=datetime.now().date()))
        ]
        
        # 添加历史消息
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 调用模型
        response = self._invoke_with_retry(messages)
        return response.content.strip()

    def chat_stream(self, session_id: str, user_input: str):
        """
        流式处理用户输入，逐 token 生成回答

        参数：
            session_id: 会话ID
            user_input: 用户输入

        Yields:
            包含 type 及对应数据的字典：
            {"type": "chunk", "content": "..."}  — 文本片段
            {"type": "complete", "answer": "...", "intent": "...", "needs_human": bool} — 完整回答（转人工等）
            {"type": "error", "answer": "..."} — 错误
            {"type": "done", "intent": "...", "needs_human": bool, "turn_count": int} — 流结束信号
        """
        # 0. 过滤输入
        user_input = sanitize_input(user_input)
        if not user_input:
            yield {"type": "complete", "answer": "请输入您的问题。", "intent": "empty_input", "needs_human": False}
            return

        # 1. 检查会话有效性
        if not self.conversation_manager.is_session_valid(session_id):
            yield {"type": "complete", "answer": "会话已过期，请重新开始对话。", "intent": "session_expired", "needs_human": False}
            return

        # 2. 保存用户消息
        self.conversation_manager.add_message(session_id, "user", user_input)

        # 3. 识别意图
        intent = self.intent_classifier.classify(user_input)

        # 4. 处理转人工请求（不需要流式）
        if intent == "human_handoff":
            answer = Config.UNKNOWN_ANSWER_TEMPLATE
            self.conversation_manager.add_message(session_id, "assistant", answer)
            yield {"type": "complete", "answer": answer, "intent": intent, "needs_human": True}
            return

        # 5. 构建消息
        try:
            if intent == "general_chat":
                messages = self._build_chat_messages(session_id)
            else:
                search_results = self.knowledge_base.search(user_input)
                if not search_results:
                    answer = Config.UNKNOWN_ANSWER_TEMPLATE
                    self.conversation_manager.add_message(session_id, "assistant", answer)
                    yield {"type": "complete", "answer": answer, "intent": intent, "needs_human": True}
                    return
                messages = self._build_rag_messages(session_id, user_input, search_results)
        except Exception as e:
            logger.error(f"构建消息失败：{e}")
            yield {"type": "error", "answer": "抱歉，处理您的问题时出现错误，请重试。"}
            return

        # 6. 流式生成
        full_answer = ""
        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "chunk", "content": chunk.content}
        except Exception as e:
            logger.error(f"流式生成失败：{e}")
            yield {"type": "error", "answer": "抱歉，生成回答时出现错误，请稍后重试。"}
            return

        # 7. 保存完整回答
        self.conversation_manager.add_message(session_id, "assistant", full_answer)

        # 8. 发送完成信号
        needs_human = (
            ("无法" in full_answer or "不知道" in full_answer) and
            intent != "general_chat"
        )
        yield {
            "type": "done",
            "intent": intent,
            "needs_human": needs_human,
            "turn_count": self.conversation_manager.sessions[session_id]["turn_count"]
        }

    def _build_chat_messages(self, session_id: str) -> list:
        """构建闲聊模式的消息列表（复用逻辑）"""
        history = self.conversation_manager.get_history(
            session_id, max_turns=Config.MAX_CONVERSATION_TURNS
        )
        messages = [
            SystemMessage(content=PromptTemplates.SYSTEM_PROMPT.format(
                current_date=datetime.now().date()
            ))
        ]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _build_rag_messages(self, session_id: str, user_input: str,
                            search_results: list) -> list:
        """构建RAG模式的消息列表（复用逻辑）"""
        context = "\n\n".join([
            f"[相关度{r['score']:.2f}] {r['content']}"
            for r in search_results
        ])

        history = self.conversation_manager.get_history(session_id, max_turns=3)
        history_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
            for m in history[:-1]
        ])

        prompt = PromptTemplates.RAG_PROMPT.format(
            context=context,
            question=user_input
        )
        if history_text:
            prompt = f"历史对话：\n{history_text}\n\n{prompt}"

        return [
            SystemMessage(content=PromptTemplates.SYSTEM_PROMPT.format(
                current_date=datetime.now().date()
            )),
            HumanMessage(content=prompt)
        ]
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息
        
        参数：
            session_id: 会话ID
        
        返回：
            包含会话状态、消息数量、轮数等信息的字典
        """
        if session_id not in self.conversation_manager.sessions:
            return {"error": "会话不存在"}
        
        session = self.conversation_manager.sessions[session_id]
        return {
            "session_id": session_id,
            "is_valid": self.conversation_manager.is_session_valid(session_id),
            "message_count": len(session["messages"]),
            "turn_count": session["turn_count"],
            "last_active": session["last_active"].isoformat()
        }


# ============================================
# 模块测试
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("智能客服系统测试")
    print("=" * 60)
    
    # 初始化机器人
    bot = CustomerServiceBot()
    
    # 创建会话
    session_id = bot.create_session()
    print(f"\n创建会话：{session_id}")
    
    # 测试对话
    test_inputs = [
        "你好",
        "智能客服系统Pro多少钱？",
        "支持哪些功能？",
        "如何申请退款？",
        "你们公司在哪里？",  # 这个问题知识库没有
        "转人工"
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"用户：{user_input}")
        print("-" * 60)
        
        result = bot.chat(session_id, user_input)
        
        print(f"助手：{result['answer']}")
        print(f"意图：{result['intent']}")
        print(f"需要转人工：{result['needs_human']}")
    
    # 打印会话信息
    print(f"\n{'='*60}")
    print("会话统计：")
    info = bot.get_session_info(session_id)
    print(json.dumps(info, ensure_ascii=False, indent=2))
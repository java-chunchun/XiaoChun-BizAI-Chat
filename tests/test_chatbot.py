# ============================================
# 智能客服系统单元测试
# 测试对话管理、意图识别、知识库检索等核心功能
# 运行命令: pytest tests/test_chatbot.py -v
# ============================================

import sys
import os
import pytest
from datetime import datetime, timedelta

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config, PromptTemplates
from knowledge_base import KnowledgeBase
from chatbot import ConversationManager, IntentClassifier, CustomerServiceBot


# ============================================
# 测试配置模块
# ============================================
class TestConfig:
    """测试配置类"""
    
    def test_config_values(self):
        """测试配置项是否正确设置"""
        assert Config.TEMPERATURE == 0.3
        assert Config.RETRIEVAL_TOP_K == 3
        assert Config.SIMILARITY_THRESHOLD == 0.7
        assert Config.SESSION_TIMEOUT == 1800
    
    def test_config_to_dict(self):
        """测试配置转字典功能"""
        config_dict = Config.to_dict()
        assert isinstance(config_dict, dict)
        assert "TEMPERATURE" in config_dict
        assert "OPENAI_API_KEY" in config_dict


# ============================================
# 测试对话管理器
# ============================================
class TestConversationManager:
    """测试对话管理器"""
    
    @pytest.fixture
    def manager(self):
        """创建对话管理器实例"""
        return ConversationManager()
    
    def test_create_session(self, manager):
        """测试创建会话"""
        session_id = manager.create_session()
        assert session_id.startswith("session_")
        assert session_id in manager.sessions
    
    def test_add_message(self, manager):
        """测试添加消息"""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "你好")
        
        history = manager.get_history(session_id)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
    
    def test_add_message_invalid_session(self, manager):
        """测试向不存在的会话添加消息"""
        with pytest.raises(ValueError):
            manager.add_message("invalid_id", "user", "你好")
    
    def test_get_history_with_limit(self, manager):
        """测试获取有限轮数的历史"""
        session_id = manager.create_session()
        
        # 添加6条消息（3轮对话）
        for i in range(3):
            manager.add_message(session_id, "user", f"问题{i}")
            manager.add_message(session_id, "assistant", f"回答{i}")
        
        # 只获取最近2轮
        history = manager.get_history(session_id, max_turns=2)
        assert len(history) == 4  # 2轮 = 4条消息
    
    def test_session_validity(self, manager):
        """测试会话有效性检查"""
        session_id = manager.create_session()
        assert manager.is_session_valid(session_id) is True
    
    def test_cleanup_expired_sessions(self, manager):
        """测试清理过期会话"""
        session_id = manager.create_session()
        
        # 手动设置过期时间
        manager.sessions[session_id]["last_active"] = datetime.now() - timedelta(seconds=Config.SESSION_TIMEOUT + 1)
        
        count = manager.cleanup_expired_sessions()
        assert count == 1
        assert session_id not in manager.sessions


# ============================================
# 测试意图分类器
# ============================================
class TestIntentClassifier:
    """测试意图分类器"""
    
    @pytest.fixture
    def classifier(self):
        """创建意图分类器实例"""
        # 注意：此测试需要有效的API密钥
        return IntentClassifier()
    
    def test_classify_price_inquiry(self, classifier):
        """测试价格咨询意图识别"""
        intent = classifier.classify("这个产品多少钱？")
        assert intent in ["price_inquiry", "product_inquiry", "general_chat"]
    
    def test_classify_human_handoff(self, classifier):
        """测试转人工意图识别"""
        intent = classifier.classify("转人工")
        assert intent == "human_handoff"
    
    def test_classify_invalid_intent_fallback(self, classifier):
        """测试无效意图回退"""
        # 模拟返回无效意图的情况
        intent = classifier.classify("")
        assert intent in ["general_chat", "human_handoff"]


# ============================================
# 测试知识库
# ============================================
class TestKnowledgeBase:
    """测试知识库模块"""
    
    @pytest.fixture
    def kb(self):
        """创建知识库实例并初始化"""
        knowledge_base = KnowledgeBase()
        knowledge_base.initialize()
        return knowledge_base
    
    def test_initialization(self, kb):
        """测试知识库初始化"""
        stats = kb.get_stats()
        assert stats["status"] == "已初始化"
        assert stats["vector_count"] > 0
    
    def test_search(self, kb):
        """测试检索功能"""
        results = kb.search("智能客服系统", top_k=2)
        assert isinstance(results, list)
        # 可能返回0个或多个结果，取决于数据
        if results:
            assert "content" in results[0]
            assert "score" in results[0]
            assert results[0]["score"] >= Config.SIMILARITY_THRESHOLD
    
    def test_search_no_results(self, kb):
        """测试无结果检索"""
        results = kb.search("完全不相关的内容xyz123")
        assert isinstance(results, list)
    
    def test_add_document(self, kb):
        """测试动态添加文档"""
        initial_count = kb.get_stats()["vector_count"]
        
        kb.add_document(
            content="测试文档内容",
            metadata={"source": "test", "title": "测试"}
        )
        
        new_count = kb.get_stats()["vector_count"]
        assert new_count >= initial_count
    
    def test_get_stats(self, kb):
        """测试获取统计信息"""
        stats = kb.get_stats()
        assert "status" in stats
        assert "vector_count" in stats
        assert "db_path" in stats


# ============================================
# 测试客服机器人
# ============================================
class TestCustomerServiceBot:
    """测试客服机器人"""
    
    @pytest.fixture
    def bot(self):
        """创建客服机器人实例"""
        return CustomerServiceBot()
    
    def test_create_session(self, bot):
        """测试创建会话"""
        session_id = bot.create_session()
        assert session_id.startswith("session_")
    
    def test_chat_welcome(self, bot):
        """测试问候语"""
        session_id = bot.create_session()
        result = bot.chat(session_id, "你好")
        
        assert "answer" in result
        assert "intent" in result
        assert "needs_human" in result
        assert result["intent"] == "general_chat"
        assert result["needs_human"] is False
    
    def test_chat_product_inquiry(self, bot):
        """测试产品咨询"""
        session_id = bot.create_session()
        result = bot.chat(session_id, "智能客服系统Pro多少钱？")
        
        assert "answer" in result
        assert result["intent"] in ["product_inquiry", "price_inquiry"]
    
    def test_chat_human_handoff(self, bot):
        """测试转人工"""
        session_id = bot.create_session()
        result = bot.chat(session_id, "转人工")
        
        assert result["intent"] == "human_handoff"
        assert result["needs_human"] is True
    
    def test_chat_expired_session(self, bot):
        """测试过期会话"""
        result = bot.chat("invalid_session", "你好")
        assert result["intent"] == "session_expired"
    
    def test_get_session_info(self, bot):
        """测试获取会话信息"""
        session_id = bot.create_session()
        info = bot.get_session_info(session_id)
        
        assert info["session_id"] == session_id
        assert info["is_valid"] is True
        assert info["message_count"] == 0
        assert info["turn_count"] == 0
    
    def test_conversation_flow(self, bot):
        """测试完整对话流程"""
        session_id = bot.create_session()
        
        # 第一轮：问候
        result1 = bot.chat(session_id, "你好")
        assert result1["turn_count"] == 1
        
        # 第二轮：咨询
        result2 = bot.chat(session_id, "有什么产品？")
        assert result2["turn_count"] == 2
        
        # 验证历史记录
        info = bot.get_session_info(session_id)
        assert info["message_count"] == 4  # 2轮 = 4条消息
        assert info["turn_count"] == 2


# ============================================
# 测试提示词模板
# ============================================
class TestPromptTemplates:
    """测试提示词模板"""
    
    def test_system_prompt(self):
        """测试系统提示词"""
        assert "客服助手" in PromptTemplates.SYSTEM_PROMPT
        assert "{current_date}" in PromptTemplates.SYSTEM_PROMPT
    
    def test_rag_prompt(self):
        """测试RAG提示词"""
        assert "{context}" in PromptTemplates.RAG_PROMPT
        assert "{question}" in PromptTemplates.RAG_PROMPT
    
    def test_intent_prompt(self):
        """测试意图识别提示词"""
        assert "{input}" in PromptTemplates.INTENT_PROMPT
        assert "product_inquiry" in PromptTemplates.INTENT_PROMPT


# ============================================
# 性能测试
# ============================================
class TestPerformance:
    """性能测试"""
    
    def test_response_time(self):
        """测试响应时间（需要API密钥）"""
        # 此测试需要实际调用API，默认跳过
        pytest.skip("需要API密钥，跳过性能测试")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])

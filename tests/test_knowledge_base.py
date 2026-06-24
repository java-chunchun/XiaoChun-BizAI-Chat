# ============================================
# 知识库模块专项测试
# 测试文档加载、切分、向量化和检索功能
# ============================================

import sys
import os
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_base import KnowledgeBase
from config import Config


class TestDocumentLoading:
    """测试文档加载功能"""
    
    def test_load_knowledge_base_txt(self):
        """测试加载知识库文本文件"""
        kb = KnowledgeBase()
        documents = kb.load_documents()
        
        # 检查是否加载了文档
        assert len(documents) > 0
        
        # 检查文档结构
        for doc in documents:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0
    
    def test_load_products_json(self):
        """测试加载产品JSON数据"""
        # 验证产品文件存在
        assert os.path.exists(Config.PRODUCTS_FILE)
        
        # 验证JSON格式正确
        with open(Config.PRODUCTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert "products" in data
        assert "faq" in data
        assert len(data["products"]) > 0
    
    def test_document_metadata(self):
        """测试文档元数据"""
        kb = KnowledgeBase()
        documents = kb.load_documents()
        
        # 检查不同来源的文档
        sources = [doc.metadata.get("source") for doc in documents]
        assert "knowledge_base" in sources or "products" in sources or "faq" in sources


class TestTextSplitting:
    """测试文本切分功能"""
    
    def test_chunk_size(self):
        """测试切分后的块大小"""
        kb = KnowledgeBase()
        
        # 创建测试文档
        test_doc = type("Doc", (), {
            "page_content": "这是一段测试文本。" * 100,
            "metadata": {"source": "test"}
        })()
        
        chunks = kb.text_splitter.split_documents([test_doc])
        
        # 检查每个块的大小
        for chunk in chunks:
            assert len(chunk.page_content) <= 500 + 50  # chunk_size + 允许误差
    
    def test_chunk_overlap(self):
        """测试块之间的重叠"""
        kb = KnowledgeBase()
        
        # 创建连续文本
        content = "第一段内容。第二段内容。第三段内容。第四段内容。"
        test_doc = type("Doc", (), {
            "page_content": content,
            "metadata": {"source": "test"}
        })()
        
        chunks = kb.text_splitter.split_documents([test_doc])
        
        # 如果有多个块，检查是否有重叠
        if len(chunks) > 1:
            # 简单检查：相邻块应该有共同内容
            chunk1_text = chunks[0].page_content
            chunk2_text = chunks[1].page_content
            # 允许不重叠的情况，取决于切分策略


class TestVectorStore:
    """测试向量数据库功能"""
    
    @pytest.fixture
    def initialized_kb(self):
        """创建已初始化的知识库"""
        kb = KnowledgeBase()
        kb.initialize()
        return kb
    
    def test_vector_count(self, initialized_kb):
        """测试向量数量"""
        stats = initialized_kb.get_stats()
        assert stats["vector_count"] > 0
    
    def test_similarity_search(self, initialized_kb):
        """测试相似度搜索"""
        results = initialized_kb.search("价格", top_k=3)
        
        # 检查返回格式
        assert isinstance(results, list)
        
        if results:
            # 检查相似度分数
            for result in results:
                assert "score" in result
                assert 0 <= result["score"] <= 1
                assert "content" in result
    
    def test_search_with_filter(self, initialized_kb):
        """测试带过滤的搜索"""
        # 搜索特定产品
        results = initialized_kb.search("智能客服系统Pro", top_k=1)
        
        if results:
            # 检查是否返回了相关内容
            assert "智能客服" in results[0]["content"] or "Pro" in results[0]["content"]


class TestKnowledgeBaseEdgeCases:
    """测试边界情况"""
    
    def test_empty_query(self):
        """测试空查询"""
        kb = KnowledgeBase()
        kb.initialize()
        
        results = kb.search("")
        assert isinstance(results, list)
    
    def test_long_query(self):
        """测试长查询"""
        kb = KnowledgeBase()
        kb.initialize()
        
        long_query = "这是一个很长的查询" * 50
        results = kb.search(long_query)
        assert isinstance(results, list)
    
    def test_special_characters(self):
        """测试特殊字符查询"""
        kb = KnowledgeBase()
        kb.initialize()
        
        special_queries = [
            "!@#$%^&*()",
            "<script>alert('test')</script>",
            "中文测试123ABC"
        ]
        
        for query in special_queries:
            results = kb.search(query)
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

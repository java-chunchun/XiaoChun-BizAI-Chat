# ============================================
# 知识库管理模块
# 负责文档加载、向量化、存储和检索
# ============================================

import json
import logging
import os as _os
from typing import List, Dict, Any, Optional
from pathlib import Path

# ── 国内网络优化：模型已缓存就走离线，否则走镜像 ──
_cache_root = Path(_os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_model_cache = _cache_root / "hub" / "models--BAAI--bge-small-zh-v1.5"
if (_model_cache / "snapshots").exists() and any((_model_cache / "snapshots").iterdir()):
    _os.environ["HF_HUB_OFFLINE"] = "1"
elif "HF_ENDPOINT" not in _os.environ:
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import OpenAIEmbeddings

# 兼容不同版本的 Chroma 导入路径
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# 兼容不同版本的 langchain 导入路径
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    知识库管理类
    
    功能：
    1. 加载各种格式的文档（txt、json、pdf等）
    2. 将文档切分成小块（chunk）
    3. 使用Embedding模型将文本转换为向量
    4. 存储到向量数据库（ChromaDB）
    5. 根据用户问题检索最相关的文档
    
    使用示例：
        kb = KnowledgeBase()
        kb.initialize()  # 初始化并加载数据
        results = kb.search("产品价格是多少？")  # 检索
    """
    
    def __init__(self):
        """
        初始化知识库
        设置文本切分器等组件（Embedding 模型延迟加载）
        """
        # 文本切分器
        # chunk_size: 每个文本块的最大字符数
        # chunk_overlap: 相邻块之间的重叠字符数，保证上下文连贯
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # 每个块500字符
            chunk_overlap=50,    # 重叠50字符
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # Embedding 模型 — 延迟到 initialize() 时才加载
        # 避免仅仅加载文档时就触发模型下载
        self.embeddings = None

        # 向量数据库实例
        self.vector_store: Optional[Chroma] = None

        logger.info("知识库已创建（Embedding 模型将在首次使用时加载）")

    def _get_embeddings(self):
        """
        延迟加载 Embedding 模型

        Embedding 方案选择：
        - local（默认）: 使用 HuggingFace 本地模型，首次运行会自动下载（约 95MB）
        - openai: 使用 OpenAI 兼容 API
        """
        if self.embeddings is not None:
            return self.embeddings

        if Config.EMBEDDING_PROVIDER == "local":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                logger.info(
                    f"正在加载本地 Embedding 模型：{Config.LOCAL_EMBEDDING_MODEL}"
                    f"（首次运行需下载，约 95MB，请稍候...）"
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=Config.LOCAL_EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu", "local_files_only": True}
                )
                logger.info("本地 Embedding 模型加载完成")
            except ImportError:
                logger.error(
                    "本地 Embedding 需要 langchain-huggingface，"
                    "请运行: pip install langchain-huggingface sentence-transformers"
                )
                raise
        else:
            logger.info(f"使用远程 Embedding 模型：{Config.EMBEDDING_MODEL}")
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                openai_api_base=Config.OPENAI_BASE_URL
            )

        return self.embeddings
    
    def load_documents(self) -> List[Document]:
        """
        加载所有数据源中的文档
        
        支持的数据源：
        1. knowledge_base.txt - 产品手册文本
        2. products.json - 产品信息JSON
        3. FAQ数据
        
        返回：
            Document对象列表，每个Document包含page_content和metadata
        """
        documents = []
        
        # 1. 加载知识库文本文件
        kb_path = Path(Config.KNOWLEDGE_BASE_FILE)
        if kb_path.exists():
            with open(kb_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 按产品分割文档
            # 每个产品作为一个独立文档
            products = content.split("【")[1:]  # 分割符是【产品名称】
            for product in products:
                if "】" in product:
                    title = product.split("】")[0]
                    content = product.split("】", 1)[1] if "】" in product else product
                    documents.append(Document(
                        page_content=content.strip(),
                        metadata={"source": "knowledge_base", "title": title}
                    ))
            logger.info(f"已加载知识库文本：{len(products)}个产品")
        
        # 2. 加载产品JSON数据
        products_path = Path(Config.PRODUCTS_FILE)
        if products_path.exists():
            with open(products_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 将每个产品转换为文档
            for product in data.get("products", []):
                # 构建产品描述文本
                content = (
                    f"产品名称：{product['name']}\n"
                    f"产品类别：{product['category']}\n"
                    f"价格：{product['price']}元\n"
                    f"产品描述：{product['description']}\n"
                    f"核心功能：{', '.join(product['features'])}\n"
                    f"服务时间：{product['support_hours']}\n"
                    f"保修政策：{product['warranty']}"
                )
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": "products",
                        "product_id": product["id"],
                        "product_name": product["name"]
                    }
                ))
            
            # 加载FAQ
            for faq in data.get("faq", []):
                content = f"问题：{faq['question']}\n答案：{faq['answer']}"
                documents.append(Document(
                    page_content=content,
                    metadata={"source": "faq", "question": faq["question"]}
                ))
            
            logger.info(f"已加载产品数据：{len(data.get('products', []))}个产品，{len(data.get('faq', []))}个FAQ")
        
        return documents
    
    def initialize(self) -> None:
        """
        初始化知识库
        
        执行流程：
        1. 加载所有文档
        2. 切分成小块
        3. 向量化并存储到ChromaDB
        
        注意：如果数据库已存在，会复用已有数据
        """
        # 检查是否已有持久化的向量数据库
        db_path = Path(Config.VECTOR_DB_PATH)
        if db_path.exists() and any(db_path.iterdir()):
            logger.info("发现已有向量数据库，正在加载...")
            self.vector_store = Chroma(
                persist_directory=Config.VECTOR_DB_PATH,
                embedding_function=self._get_embeddings()
            )
            logger.info("向量数据库加载完成")
            return
        
        # 加载文档
        logger.info("开始加载文档...")
        documents = self.load_documents()
        
        if not documents:
            logger.warning("没有找到任何文档！")
            return
        
        # 切分文档
        logger.info("正在切分文档...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"文档切分完成：{len(documents)}个文档 → {len(chunks)}个块")
        
        # 创建向量数据库
        logger.info("正在创建向量数据库（首次初始化需要几分钟）...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            persist_directory=Config.VECTOR_DB_PATH
        )
        
        # 持久化保存
        self.vector_store.persist()
        logger.info("向量数据库创建并保存完成")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        根据用户问题检索最相关的文档
        
        参数：
            query: 用户输入的问题
            top_k: 返回的最相似文档数量，默认使用配置中的值
        
        返回：
            包含文档内容和相似度分数的列表
            格式：[{"content": "文本内容", "score": 0.95, "metadata": {...}}, ...]
        
        使用示例：
            results = kb.search("智能客服系统的价格是多少？")
            for r in results:
                print(f"相似度：{r['score']:.2f}")
                print(f"内容：{r['content']}")
        """
        if self.vector_store is None:
            raise RuntimeError("知识库未初始化，请先调用initialize()")

        # 使用配置中的默认值
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
        
        logger.info(f"正在检索：'{query}'，返回{top_k}个结果")
        
        # 执行相似度搜索
        # 返回文档和分数
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # 格式化结果
        formatted_results = []
        for doc, score in results:
            # 过滤低相似度结果
            if score < Config.SIMILARITY_THRESHOLD:
                continue
            
            formatted_results.append({
                "content": doc.page_content,
                "score": round(score, 4),
                "metadata": doc.metadata
            })
        
        logger.info(f"检索完成，找到{len(formatted_results)}个相关结果")
        return formatted_results
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        动态添加新文档到知识库
        
        参数：
            content: 文档内容
            metadata: 文档元数据（如来源、标题等）
        
        使用场景：
            当有新知识需要补充时，无需重新初始化整个知识库
        """
        if self.vector_store is None:
            raise RuntimeError("知识库未初始化")
        
        # 创建Document对象
        doc = Document(page_content=content, metadata=metadata or {})
        
        # 切分文档
        chunks = self.text_splitter.split_documents([doc])
        
        # 添加到向量数据库
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        
        logger.info(f"已添加新文档：{len(chunks)}个块")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        返回：
            包含文档数量、存储路径等信息的字典
        """
        if self.vector_store is None:
            return {"status": "未初始化"}
        
        return {
            "status": "已初始化",
            "vector_count": self.vector_store._collection.count(),
            "db_path": Config.VECTOR_DB_PATH,
            "embedding_model": Config.EMBEDDING_MODEL
        }


# ============================================
# 模块测试
# ============================================
if __name__ == "__main__":
    # 测试知识库功能
    print("=" * 50)
    print("知识库模块测试")
    print("=" * 50)
    
    kb = KnowledgeBase()
    kb.initialize()
    
    # 打印统计信息
    stats = kb.get_stats()
    print(f"\n知识库状态：{stats}")
    
    # 测试检索
    test_queries = [
        "智能客服系统的价格是多少？",
        "如何申请退款？",
        "云存储服务支持哪些功能？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"测试查询：{query}")
        print("=" * 50)
        
        results = kb.search(query)
        for i, r in enumerate(results, 1):
            print(f"\n结果{i}（相似度：{r['score']:.2f}）：")
            print(r['content'][:200] + "...")  # 只显示前200字符
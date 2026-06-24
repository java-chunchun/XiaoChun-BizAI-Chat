# LangChain 企业级智能客服系统

## TL;DR

一个生产可用的 RAG 智能客服系统，5 分钟部署，**国内直连，无需 VPN**。

- 🧠 DeepSeek / 智谱 / GPT-4 多模型切换，默认 DeepSeek（¥2/百万 token）
- 📚 本地 Embedding（免费）+ ChromaDB 向量检索
- 💬 流式输出 + 多轮对话上下文管理
- 🎯 6 类意图自动识别 + 智能转人工
- 🌐 FastAPI HTTP API + Swagger 文档
- 🐳 Docker Compose 一键部署
- ✅ 完整测试覆盖 + 输入安全过滤 + LLM 调用重试

## 一、功能清单

### 1. RAG 检索增强生成

- 本地 BGE Embedding 模型向量化知识库（完全免费，无需 API）
- ChromaDB 语义检索 + 相似度阈值过滤（默认 0.7）
- 检索结果注入 LLM 上下文，大幅提升回答准确度
- 知识库命中不足时自动降级为友好兜底回复

### 2. 6 类意图自动识别

| 意图 | 标识 | 处理策略 |
|------|------|---------|
| 产品咨询 | `product_inquiry` | RAG 检索增强 |
| 价格咨询 | `price_inquiry` | RAG 检索增强 |
| 技术支持 | `technical_support` | RAG 检索增强 |
| 投诉反馈 | `complaint` | RAG + 转人工判断 |
| 闲聊 | `general_chat` | 直接 LLM 对话 |
| 转人工 | `human_handoff` | 直接触发转人工 |

### 3. 多轮对话管理

- 会话生命周期管理（创建 → 活跃 → 超时自动清理）
- 上下文窗口限制（可配轮数，防止 token 爆炸）
- 对话历史注入 LLM，解决指代消解问题
- 30 分钟无活动自动过期，释放内存

### 4. 流式输出

- 逐 token 实时推送，首字延迟 < 1s
- CLI 模式下实时打印，体感响应极快
- 支持 `chat_stream()` 生成器接口，可对接 SSE

### 5. 智能转人工

- 回答中包含"无法"/"不知道"关键词 → 自动标记转人工
- 检索相似度全部低于阈值 → 建议转人工
- 用户明确要求"转人工" → 直接触发

### 6. HTTP API 服务

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/session/new` | POST | 创建新会话 |
| `/chat` | POST | 发送消息，返回回复 + 意图 + 转人工标记 |
| `/session/{id}/info` | GET | 查看会话详情 |
| `/docs` | GET | Swagger 交互式 API 文档 |
| `/redoc` | GET | ReDoc API 文档 |

### 7. 工程化保障

- **输入安全过滤**：控制字符清洗 + 超长截断（2000 字），防注入
- **LLM 调用重试**：最多 2 次指数退避重试（2s → 4s），失败兜底友好提示
- **请求超时控制**：意图分类 15s，回答生成 60s
- **配置校验**：启动时检查 API 密钥、数据文件、依赖完整性

---

## 二、技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| AI 框架 | LangChain 1.x | Agent 编排、Chain 构建、Memory 管理 |
| 大语言模型 | DeepSeek / 智谱 GLM-4 / GPT-4 | OpenAI 兼容协议，一键切换 |
| Embedding | BAAI/bge-small-zh-v1.5（本地） | 中文优化，512 维，95MB，CPU 即可 |
| 向量数据库 | ChromaDB | 本地持久化，重启无需重建 |
| 文本切分 | RecursiveCharacterTextSplitter | chunk=500 / overlap=50，中文标点优先 |
| Web 框架 | FastAPI + uvicorn | 异步高性能，自带 Swagger 文档 |
| 数据校验 | Pydantic v2 | 请求/响应模型自动校验 |
| 容器化 | Docker + Docker Compose | 一键构建部署 |
| 测试 | pytest + pytest-cov | 单元测试 + 覆盖率报告 |
| LLM 客户端 | httpx | 异步 HTTP，支持重试与超时 |
| 模型推理 | sentence-transformers 3.x | 本地 Embedding 推理引擎 |

---

## 三、项目亮点

### 🏗️ 架构设计

1. **模块化分层清晰**：`config`（配置）→ `knowledge_base`（知识库）→ `chatbot`（对话）→ `api`（服务），职责分明，低耦合
2. **依赖倒置**：LLM 和 Embedding 均通过配置切换，不绑定特定厂商
3. **懒加载设计**：Embedding 模型只在首次向量化时加载，文档扫描不触发 95MB 模型下载

### 💰 成本控制

| 对比维度 | 本项目（DeepSeek + 本地 Embedding） | 传统方案（GPT-4 + 远程 Embedding） |
|---------|----------------------------------|----------------------------------|
| LLM 输入 | ¥2 / 百万 token | ¥110 / 百万 token |
| LLM 输出 | ¥8 / 百万 token | ¥220 / 百万 token |
| Embedding | **免费**（本地 CPU） | $0.10 / 千次 |
| 日 200 次对话 | ≈ ¥2 | ≈ ¥200 |
| VPN 需求 | 不需要 | 需要 |

### 🌐 国内网络适配

1. **LLM 直连**：DeepSeek API 国内直接访问，零延迟
2. **Embedding 本地化**：模型首次下载后完全离线运行，不依赖外网
3. **HuggingFace 智能降级**：检测模型缓存 → 离线加载 / 未缓存 → 镜像下载，自动适配网络环境

### 🧠 上下文控制

长对话场景下，核心矛盾是：**历史太少 → 丢失指代**，**历史太多 → token 爆炸 + 注意力稀释**。系统采用分层窗口策略：

| 路径 | 窗口大小 | 逻辑 |
|------|---------|------|
| RAG（业务咨询） | 最近 **3 轮** | 检索文档已占大量 token，历史不宜过多 |
| 闲聊（general_chat） | 最近 **10 轮** | 闲聊依赖上下文理解指代，窗口可适当放宽 |
| 兜底（human_handoff） | **不注入历史** | 直接返回转人工引导，无需上下文 |
| 会话生命周期 | **30 分钟**超时 | 到期自动清理，释放内存 |

> `MAX_CONVERSATION_TURNS` 同时控制闲聊路径的窗口上限，超出部分自动截断丢弃。

### 🛡️ 鲁棒性设计

1. **版本兼容导入**：try/except 兼容 LangChain 0.2.x ~ 1.x 的模块路径变更
2. **意图分类回退**：分类失败自动回退到 `general_chat`，不阻塞对话
3. **回答生成兜底**：生成失败返回友好提示 + 人工联系方式
4. **会话过期处理**：过期会话自动返回引导，不抛异常

### 📊 知识库设计

1. **多源数据加载**：txt 产品手册 + JSON 结构化数据 + FAQ 自动拼接
2. **中文友好切分**：优先按中文标点（。！？）分隔，保持语义完整
3. **增量更新**：`add_document()` 方法支持运行时动态追加知识

---

## 四、开发难点与解决方案

### 难点 1：Chroma 版本兼容 —— `__bool__` 陷阱

**现象**：向量数据库初始化成功（日志显示"加载完成"），但检索时抛出"知识库未初始化"。

**根因**：代码使用 `if not self.vector_store` 判空。`langchain_community.vectorstores.Chroma` 对象在 LangChain 1.x + ChromaDB 1.5 环境下，其 `__bool__` 方法返回 `False`（某些版本中集合为空或元数据不匹配时），导致已初始化的对象被误判为 `None`。

**解决**：将所有 `if not self.vector_store` 改为 `if self.vector_store is None`，精确判断对象是否为 `None` 而非依赖隐式布尔转换。

```python
# ❌ 错误写法——Chroma 对象可能为 Falsy
if not self.vector_store:
    raise RuntimeError("知识库未初始化")

# ✅ 正确写法
if self.vector_store is None:
    raise RuntimeError("知识库未初始化")
```

### 难点 2：HuggingFace 国内网络超时

**现象**：加载本地 Embedding 模型时，`sentence-transformers` 反复尝试连接 `huggingface.co`，每次重试 5 轮、最长等待 8s × 5 = 40s，导致启动卡死。

**根因**：`sentence-transformers` 3.x 在加载模型时会通过 `transformers` → `huggingface_hub` 发起 HTTP HEAD 请求校验 `adapter_config.json` 等配置文件，即使模型已完整缓存也无法跳过。环境变量 `HF_HUB_OFFLINE` 被设置后，底层 `httpx` 客户端被关闭，后续代码路径仍尝试发送请求，抛出 `RuntimeError: Cannot send a request, as the client has been closed.`

**解决**：在 `HuggingFaceEmbeddings` 构造时显式传入 `local_files_only=True`，直接告诉 `transformers` 库不要发起任何网络请求。同时在模块顶层检测模型缓存状态：已缓存则设 `HF_HUB_OFFLINE=1`，未缓存则设 `HF_ENDPOINT=https://hf-mirror.com`。

```python
# 模块加载时（在 import sentence_transformers 之前）
_cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_model_cache = _cache_root / "hub" / "models--BAAI--bge-small-zh-v1.5"
if (_model_cache / "snapshots").exists() and any((_model_cache / "snapshots").iterdir()):
    os.environ["HF_HUB_OFFLINE"] = "1"       # 已缓存 → 完全离线
elif "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 未缓存 → 国内镜像

# 构造时显式禁用网络
self.embeddings = HuggingFaceEmbeddings(
    model_name=Config.LOCAL_EMBEDDING_MODEL,
    model_kwargs={"device": "cpu", "local_files_only": True}
)
```

### 难点 3：ChromaDB 向量库跨版本不兼容

**现象**：项目在不同时间点使用不同版本的 ChromaDB 构建向量库，升级后旧库加载成功但集合为空（`collection.count() == 0`），检索无结果。

**根因**：ChromaDB 在不同版本之间修改了底层存储格式（SQLite schema 变更、Embedding 维度元数据不兼容），旧版本的持久化文件在新版本加载后表现为空集合，且无明确报错。

**解决**：
1. 运行时检测 `collection.count() == 0`，自动触发重建
2. 部署文档中增加故障排查条目：删除 `data/vector_db` 目录后重新启动
3. Docker 部署时将向量库持久化到 named volume，升级时通过环境变量控制是否重建

### 难点 4：LangChain 1.x 大版本迁移

**现象**：LangChain 从 0.2 升级到 1.x 后，大量模块路径变更，`langchain_community.vectorstores.Chroma` 被标记废弃。

**根因**：LangChain 1.0 将社区组件拆分到独立包（`langchain-chroma`、`langchain-text-splitters` 等），导入路径从 `langchain.xxx` 变为 `langchain_core.xxx`，且部分类 API 发生变化。

**解决**：所有导入使用 try/except 兼容新旧两种路径，确保同一套代码在 LangChain 0.2.x 和 1.x 下均可运行。

```python
# Chroma 兼容
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Messages 兼容
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage
```

### 难点 5：知识库检索零结果场景处理

**现象**：用户问题与知识库语义差距大时（如"你们公司在哪"），相似度全部低于阈值，检索返回空列表。

**根因**：`similarity_search_with_score` 返回的是余弦距离（值域 [0, 2]），而过滤条件使用 `score < SIMILARITY_THRESHOLD`（默认 0.7），将距离当作相似度使用，导致逻辑语义偏差。

**解决**：
1. 检索结果为空时返回兜底回复模板（人工联系方式）
2. 结合意图识别：`human_handoff` 意图直接跳过检索，返回转人工引导
3. 业务咨询意图（product/price/technical/complaint）检索为空时标记 `needs_human=True`

### 难点 6：命令注入与输入安全

**现象**：用户输入可能包含控制字符、超长文本、或恶意提示词注入。

**解决**：
1. **控制字符清洗**：正则过滤 `\x00-\x08\x0b\x0c\x0e-\x1f\x7f` 范围的控制字符
2. **超长截断**：默认 2000 字硬截断，防止 token 耗尽
3. **Pydantic 校验**：API 层通过 `Field(min_length=1, max_length=2000)` 二次把关
4. **独立 System Prompt**：系统指令与用户输入分离，降低提示词注入风险

---

## 五、项目结构

```
8.LangChain智能客服系统/
├── src/                          # 源代码
│   ├── config.py                 # 配置中心 + 校验
│   ├── knowledge_base.py         # 知识库管理（Embedding、切分、检索、增量更新）
│   ├── chatbot.py                # 对话引擎（会话管理、意图分类、RAG、流式输出）
│   └── api.py                    # FastAPI 路由 + Pydantic 模型
├── data/                         # 知识库数据
│   ├── products.json             # 3 产品 + 5 FAQ
│   ├── knowledge_base.txt        # 3 份产品手册
│   └── vector_db/                # ChromaDB 持久化（自动生成）
├── tests/                        # 测试
│   ├── test_chatbot.py           # 配置 / 对话 / 意图 / 提示词测试
│   └── test_knowledge_base.py    # 文档加载 / 切分 / 检索测试
├── deploy/                       # 部署
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
├── docs/                         # 文档
│   ├── README.md                 # 项目文档（本文件）
│   └── 部署文档.md               # 部署指南
├── main.py                       # 入口（CLI 对话 / 演示 / API）
└── requirements.txt              # 依赖清单
```

---

## 六、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-你的DeepSeek密钥"

# Linux / Mac / Git Bash
export OPENAI_API_KEY="sk-你的DeepSeek密钥"
```

默认使用 DeepSeek + 本地 Embedding，无需额外配置。首次运行会自动下载 Embedding 模型（约 95MB，仅一次）。

### 3. 运行

```bash
python main.py              # CLI 交互对话
python main.py --demo       # 演示模式（6 个测试用例自动运行）
python main.py --api        # 启动 API 服务 → http://localhost:8000/docs
```

---

## 七、测试

```bash
# 不依赖 API 的测试（配置、切分、对话管理等）
pytest tests/test_chatbot.py::TestConfig \
       tests/test_chatbot.py::TestPromptTemplates \
       tests/test_chatbot.py::TestConversationManager \
       tests/test_knowledge_base.py::TestDocumentLoading \
       tests/test_knowledge_base.py::TestTextSplitting -v

# 全部测试（需要 API Key）
pytest tests/ -v

# 覆盖率报告
pytest --cov=src tests/
```

---

## 八、模型切换

```bash
# 智谱 GLM
export OPENAI_API_KEY="你的智谱密钥"
export OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export PRIMARY_MODEL="glm-4-flash"

# OpenAI GPT-4（需 VPN）
export OPENAI_API_KEY="sk-你的OpenAI密钥"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export PRIMARY_MODEL="gpt-4-turbo"
```

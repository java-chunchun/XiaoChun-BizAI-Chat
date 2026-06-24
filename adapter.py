"""
企业微信 / 微信公众号 / 钉钉 等多渠道适配器

本质：渠道消息 ←→ /chat API 的薄薄一层翻译

使用方式（以企业微信为例）：
  1. 企业微信管理后台 → 应用管理 → 自建应用 → 开启 API 接收消息
  2. 拿到 Token + EncodingAESKey + CorpID + Secret
  3. 填到下面配置，启动此服务
  4. 在企微后台填入此服务的公网 URL 作为回调地址

启动：
  python adapter.py --port 8001
"""

import json
import time
import hashlib
import logging
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adapter")

# ============================================
# 配置 — 按需修改
# ============================================
CHAT_API = "http://localhost:8000"     # 你的 /chat API 地址

# 会话缓存：渠道用户 ID → session_id
# 生产环境换 Redis，一行改动
sessions: dict[str, dict] = {}         # {user_id: {session_id, last_active}}
SESSION_TTL = 1800                     # 30 分钟过期

app = FastAPI(title="多渠道适配器", docs_url=None)


# ============================================
# 核心：渠道消息 → 标准 API → 渠道回复
# ============================================
async def handle_message(user_id: str, content: str) -> str:
    """统一的对话处理入口"""
    # 1. 获取或创建 session
    now = time.time()
    if user_id not in sessions or (now - sessions[user_id]["last_active"]) > SESSION_TTL:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{CHAT_API}/session/new")
            r.raise_for_status()
            sessions[user_id] = {
                "session_id": r.json()["session_id"],
                "last_active": now
            }

    sessions[user_id]["last_active"] = now
    session_id = sessions[user_id]["session_id"]

    # 2. 调用 /chat
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{CHAT_API}/chat",
            json={"session_id": session_id, "message": content}
        )
        r.raise_for_status()
        data = r.json()

    # 3. 如果需要转人工，追加提示
    answer = data["answer"]
    if data.get("needs_human"):
        answer += "\n\n💡 如需人工协助，请拨打 400-888-9999"

    return answer


# ============================================
# 渠道 1：企业微信 回调
# ============================================
@app.api_route("/wecom", methods=["GET", "POST"])
async def wecom_callback(request: Request,
                          msg_signature: str = Query(...),
                          timestamp: str = Query(...),
                          nonce: str = Query(...)):
    """
    企业微信接收消息回调

    前置步骤：
      1. 企微管理后台 → 自建应用 → 接收消息 → 设置回调 URL 为此地址
      2. 填入 Token / EncodingAESKey
      3. 点击验证（企微会发 GET 请求到此端点）

    GET  → 企微 URL 验证（回复 echostr）
    POST → 用户消息推送（回复加密 XML）
    """
    # 这里只写骨架，实际需要 wecom_crypto 库做加解密
    # pip install wecom-crypto
    if request.method == "GET":
        # URL 验证
        try:
            from wecom_crypto import WXBizMsgCrypt
            # 用你的 Token / EncodingAESKey / CorpID 初始化
            # crypt = WXBizMsgCrypt(TOKEN, ENCODING_AES_KEY, CORP_ID)
            # echo_str = crypt.verify_url(msg_signature, timestamp, nonce, echostr)
            # return PlainTextResponse(echo_str)
            return PlainTextResponse("verify ok — 请替换为真实加解密逻辑")
        except ImportError:
            return PlainTextResponse("pip install wecom-crypto")

    # POST：接收消息
    body = await request.body()
    # 解密 body → 提取 <Content> 和 <FromUserName>
    # reply = await handle_message(user_id, content)
    # 加密 reply → 返回 XML
    return PlainTextResponse("success")


# ============================================
# 渠道 2：通用 HTTP Webhook（任何系统都能调）
# ============================================
@app.post("/webhook/chat")
async def webhook_chat(request: Request):
    """
    通用 Webhook — 任何能发 HTTP 请求的系统都能接

    请求体：
      {"user_id": "xxx", "message": "你好"}

    返回：
      {"answer": "...", "intent": "...", "needs_human": false}
    """
    body = await request.json()
    user_id = str(body.get("user_id", "anonymous"))
    message = str(body.get("message", ""))
    if not message.strip():
        return {"error": "message 不能为空"}

    answer = await handle_message(user_id, message)
    return {"answer": answer, "user_id": user_id}


# ============================================
# 渠道 3：微信公众号 回调
# ============================================
@app.api_route("/wechat", methods=["GET", "POST"])
async def wechat_callback(request: Request,
                           signature: str = Query(""),
                           timestamp: str = Query(""),
                           nonce: str = Query(""),
                           echostr: str = Query("")):
    """
    微信公众号消息回调

    前置步骤：
      1. 公众号后台 → 基本配置 → 服务器配置
      2. 填入此 URL + 自定 Token
      3. 验证通过后，用户发消息即 POST 到这里
    """
    # 这里同样只写骨架
    if request.method == "GET":
        # 微信 URL 验证
        # token = "你自己设的 token"
        # tmp = sorted([token, timestamp, nonce])
        # if hashlib.sha1("".join(tmp).encode()).hexdigest() == signature:
        #     return PlainTextResponse(echostr)
        return PlainTextResponse(echostr)

    body = await request.body()
    # 解析 XML → 提取 <Content> 和 <FromUserName>
    # reply = await handle_message(user_id, content)
    # 构建回复 XML → 返回
    return PlainTextResponse("success")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

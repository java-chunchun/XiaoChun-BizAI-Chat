#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================
# 智能客服系统入口文件
# 提供命令行交互界面和API服务启动
# ============================================

import argparse
import sys
import os

# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.chatbot import CustomerServiceBot
from src.config import Config


def run_cli():
    """
    运行命令行交互模式
    
    用户可以在终端中与智能客服进行对话
    输入 'quit' 或 'exit' 退出对话
    """
    print("=" * 60)
    print("  LangChain 企业级智能客服系统")
    print("=" * 60)
    print()
    
    # 初始化客服机器人
    print("正在初始化系统...")
    bot = CustomerServiceBot()
    print("系统初始化完成！\n")
    
    # 创建新会话
    session_id = bot.create_session()
    print(f"会话ID: {session_id}")
    print()
    
    # 显示欢迎语
    print(f"客服: {Config.WELCOME_MESSAGE}")
    print("-" * 60)
    
    # 进入对话循环
    while True:
        # 获取用户输入
        try:
            user_input = input("\n您: ").strip()
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        
        # 检查退出命令
        if user_input.lower() in ["quit", "exit", "退出", "再见"]:
            print("\n客服: 感谢您的咨询，再见！")
            break
        
        # 检查空输入
        if not user_input:
            continue
        
        # 处理用户输入
        print(f"\n客服: ", end="", flush=True)

        needs_human = False
        intent = ""
        try:
            for chunk in bot.chat_stream(session_id, user_input):
                if chunk["type"] == "chunk":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "complete":
                    print(chunk["answer"])
                    needs_human = chunk.get("needs_human", False)
                    intent = chunk.get("intent", "")
                elif chunk["type"] == "error":
                    print(f"\n{chunk['answer']}")
                elif chunk["type"] == "done":
                    print()  # 流结束后换行
                    needs_human = chunk.get("needs_human", False)
                    intent = chunk.get("intent", "")
        except Exception as e:
            print(f"\n[错误] 对话处理失败：{e}")
            continue

        # 如果需要转人工，给出提示
        if needs_human:
            print("\n[系统提示] 当前问题可能需要人工客服处理")

        # 显示调试信息（仅在DEBUG模式下）
        if Config.DEBUG and intent:
            print(f"\n[调试] 意图: {intent}")


def run_demo():
    """
    运行演示模式
    
    自动执行预设的对话流程，展示系统功能
    适合快速验证系统是否正常工作
    """
    print("=" * 60)
    print("  智能客服系统 - 演示模式")
    print("=" * 60)
    
    # 初始化
    bot = CustomerServiceBot()
    session_id = bot.create_session()
    
    # 预设测试用例
    test_cases = [
        {
            "input": "你好",
            "description": "问候语测试"
        },
        {
            "input": "智能客服系统Pro多少钱？",
            "description": "产品价格咨询"
        },
        {
            "input": "支持哪些功能？",
            "description": "产品功能咨询"
        },
        {
            "input": "如何申请退款？",
            "description": "售后政策咨询"
        },
        {
            "input": "你们公司在哪里？",
            "description": "知识库外的问题（应触发转人工）"
        },
        {
            "input": "转人工",
            "description": "转人工请求"
        }
    ]
    
    print(f"\n会话ID: {session_id}\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(test_cases)}: {case['description']}")
        print(f"{'='*60}")
        print(f"用户: {case['input']}")
        print("-" * 60)
        
        result = bot.chat(session_id, case["input"])
        
        print(f"客服: {result['answer']}")
        print(f"\n[意图: {result['intent']}, 转人工: {result['needs_human']}]")
    
    # 显示会话统计
    print(f"\n{'='*60}")
    print("会话统计")
    print(f"{'='*60}")
    info = bot.get_session_info(session_id)
    for key, value in info.items():
        print(f"{key}: {value}")


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """
    启动API服务

    使用 FastAPI + uvicorn 提供 RESTful API
    包含 /chat、/session/new、/health 等端点

    参数：
        host: 监听地址（默认 0.0.0.0，接受所有来源）
        port: 监听端口（默认 8000）
    """
    import uvicorn
    from src.api import create_app

    print("=" * 60)
    print("  LangChain 智能客服系统 API 服务")
    print("=" * 60)
    print(f"  监听地址: http://{host}:{port}")
    print(f"  API 文档: http://{host}:{port}/docs")
    print(f"  健康检查: http://{host}:{port}/health")
    print("=" * 60)
    print()

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=Config.LOG_LEVEL.lower())


def main():
    """
    主函数
    
    解析命令行参数并执行相应模式
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="LangChain 企业级智能客服系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py              # 启动交互式对话
  python main.py --demo       # 运行演示模式
  python main.py --api        # 显示API服务说明
        """
    )
    
    # 添加参数
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示模式（自动执行测试用例）"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="启动 API 服务（需先安装 fastapi 和 uvicorn）"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API 服务监听地址（默认：0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 服务监听端口（默认：8000）"
    )

    # 解析参数
    args = parser.parse_args()

    # 根据参数执行相应模式
    if args.demo:
        run_demo()
    elif args.api:
        run_api(host=args.host, port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()

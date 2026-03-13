from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
import os
import dotenv
from pathlib import Path
from typing import Optional

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

# 全局 LLM 实例
_llm_instance: Optional[BaseChatModel] = None

def get_llm() -> BaseChatModel:
    """获取 LangChain LLM 实例（单例模式）"""
    global _llm_instance

    if _llm_instance is None:
        # 从环境变量读取配置
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL_ID")

        # 验证必要的配置
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未配置（LangChain 必需）")

        # 创建 ChatOpenAI 实例
        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.7,
            max_tokens=2000,
            timeout=60.0,
            max_retries=3
        )

        print(f"[SUCCESS] LangChain LLM 初始化成功")
        print(f"   模型: {model}")
        print(f"   Base URL: {base_url}")

    return _llm_instance

def reset_llm():
    """重置 LLM 实例（用于测试或重新配置）"""
    global _llm_instance
    _llm_instance = None

if __name__ == "__main__":
    try:
        llm = get_llm()
        print("LLM 获取成功:", llm)
    except Exception as e:
        print("LLM 获取失败:", str(e))
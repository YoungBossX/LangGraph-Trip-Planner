from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import settings

# 全局 LLM 实例
_llm_instance: Optional[BaseChatModel] = None

def get_llm() -> BaseChatModel:
    """获取 LangChain LLM 实例（单例模式）"""
    global _llm_instance

    if _llm_instance is None:
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        model = settings.llm_model_id or settings.llm_model

        if not api_key:
            raise ValueError("LLM_API_KEY未配置")
        if not base_url:
            raise ValueError("LLM_BASE_URL未配置")
        if not model:
            raise ValueError("LLM_MODEL_ID未配置")

        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            timeout=settings.agent_timeout,
            max_retries=settings.agent_max_iterations
        )

        print("[SUCCESS] LangChain LLM 初始化成功")
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

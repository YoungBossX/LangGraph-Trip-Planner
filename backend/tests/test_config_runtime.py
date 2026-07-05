from unittest.mock import MagicMock


def test_validate_config_uses_settings_values_without_process_env(monkeypatch):
    from app import config

    configured = config.Settings(
        amap_api_key="amap-key",
        llm_api_key="llm-key",
        llm_base_url="https://example.test/v1",
        llm_model_id="model-id",
    )
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    assert config.validate_config() is True


def test_get_llm_uses_settings_values_without_process_env(monkeypatch):
    from app import config
    from app.services import llm_service

    configured = config.Settings(
        amap_api_key="amap-key",
        llm_api_key="llm-key",
        llm_base_url="https://example.test/v1",
        llm_model_id="model-id",
    )
    fake_chat_openai = MagicMock(return_value=object())

    llm_service.reset_llm()
    monkeypatch.setattr(llm_service, "settings", configured)
    monkeypatch.setattr(llm_service, "ChatOpenAI", fake_chat_openai)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL_ID", raising=False)

    llm_service.get_llm()

    fake_chat_openai.assert_called_once_with(
        api_key="llm-key",
        base_url="https://example.test/v1",
        model="model-id",
        temperature=configured.agent_temperature,
        max_tokens=configured.agent_max_tokens,
        timeout=configured.agent_timeout,
        max_retries=configured.agent_max_iterations,
    )

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

GUARDRAIL_DEFAULTS = {
    "max_request_body_bytes": 16384,
    "planning_rate_limit": 3,
    "planning_rate_window_seconds": 600,
    "planning_per_ip_concurrency": 1,
    "planning_global_concurrency": 2,
    "photo_rate_limit": 30,
    "photo_rate_window_seconds": 60,
    "trip_request_timeout_seconds": 300,
    "mcp_tool_timeout_seconds": 45,
    "sse_heartbeat_seconds": 15,
}


def test_guardrail_settings_have_exact_defaults(monkeypatch):
    from app import config

    for field_name in GUARDRAIL_DEFAULTS:
        monkeypatch.delenv(field_name, raising=False)
        monkeypatch.delenv(field_name.upper(), raising=False)

    configured = config.Settings(_env_file=None)

    assert {field_name: getattr(configured, field_name) for field_name in GUARDRAIL_DEFAULTS} == GUARDRAIL_DEFAULTS


@pytest.mark.parametrize("field_name", GUARDRAIL_DEFAULTS)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_guardrail_settings_reject_non_positive_values(field_name, invalid_value):
    from app import config

    with pytest.raises(ValidationError):
        config.Settings(_env_file=None, **{field_name: invalid_value})


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

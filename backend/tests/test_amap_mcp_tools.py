import asyncio
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import StructuredTool


def _structured_async_tool(
    coroutine,
    *,
    name="async-tool",
    response_format="content",
    metadata=None,
    tags=None,
    return_direct=False,
):
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description="Async test tool",
        response_format=response_format,
        metadata=metadata,
        tags=tags,
        return_direct=return_direct,
    )


def test_amap_mcp_connection_uses_project_uv_cache(monkeypatch):
    from app import config
    from app.tools import amap_mcp_tools

    configured = config.Settings(amap_api_key="amap-key")
    monkeypatch.setattr(amap_mcp_tools, "get_settings", lambda: configured)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)

    connection = amap_mcp_tools._build_amap_mcp_connection()

    assert connection["command"] == "uvx"
    assert connection["args"] == ["--python", sys.executable, "amap-mcp-server"]
    assert connection["transport"] == "stdio"
    assert connection["env"]["AMAP_MAPS_API_KEY"] == "amap-key"
    assert connection["env"]["UV_CACHE_DIR"] == str(Path(amap_mcp_tools.__file__).resolve().parents[2] / ".uv-cache")
    assert connection["env"]["UV_TOOL_DIR"] == str(Path(amap_mcp_tools.__file__).resolve().parents[2] / ".uv-tools")
    assert connection["env"]["UV_PYTHON_DOWNLOADS"] == "never"


def test_amap_mcp_connection_respects_existing_uv_dirs(monkeypatch):
    from app import config
    from app.tools import amap_mcp_tools

    custom_cache_dir = "X:\\custom-uv-cache"
    custom_tool_dir = "X:\\custom-uv-tools"
    configured = config.Settings(amap_api_key="amap-key")
    monkeypatch.setattr(amap_mcp_tools, "get_settings", lambda: configured)
    monkeypatch.setenv("UV_CACHE_DIR", custom_cache_dir)
    monkeypatch.setenv("UV_TOOL_DIR", custom_tool_dir)

    connection = amap_mcp_tools._build_amap_mcp_connection()

    assert connection["env"]["UV_CACHE_DIR"] == custom_cache_dir
    assert connection["env"]["UV_TOOL_DIR"] == custom_tool_dir


def test_async_wrapper_times_out_and_cancels_slow_coroutine(monkeypatch):
    from app.tools import amap_mcp_tools

    cancelled = threading.Event()

    async def slow(value: str):
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(slow, name="slow-tool")])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.01),
    )

    with pytest.raises(TimeoutError):
        wrapped._run(value="input")

    assert cancelled.wait(timeout=0.2)


def test_async_wrapper_returns_normal_result(monkeypatch):
    from app.tools import amap_mcp_tools

    async def normal(value: str):
        return f"result:{value}"

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(normal)])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=1),
    )

    assert wrapped._run(value="input") == "result:input"


def test_async_wrapper_preserves_response_format_and_metadata(monkeypatch):
    from app.tools import amap_mcp_tools

    async def content_and_artifact(value: str):
        return value, {"source": "test"}

    tool = _structured_async_tool(
        content_and_artifact,
        response_format="content_and_artifact",
        metadata={"provider": "amap"},
        tags=["maps"],
        return_direct=True,
    )

    wrapped = amap_mcp_tools.wrap_async_tools([tool])[0]

    assert wrapped.response_format == "content_and_artifact"
    assert wrapped.metadata == {"provider": "amap"}
    assert wrapped.tags == ["maps"]
    assert wrapped.return_direct is True
    assert wrapped.args_schema is tool.args_schema


def _force_running_loop_fallback(monkeypatch, amap_mcp_tools, future):
    loop = MagicMock()
    loop.is_running.return_value = True

    def fail_asyncio_run(coroutine):
        coroutine.close()
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    def submit_coroutine(coroutine, submitted_loop):
        assert submitted_loop is loop
        coroutine.close()
        return future

    monkeypatch.setattr(amap_mcp_tools.nest_asyncio, "apply", lambda: None)
    monkeypatch.setattr(amap_mcp_tools.asyncio, "run", fail_asyncio_run)
    monkeypatch.setattr(amap_mcp_tools.asyncio, "get_event_loop", lambda: loop)
    monkeypatch.setattr(amap_mcp_tools.asyncio, "run_coroutine_threadsafe", submit_coroutine)


def test_running_loop_fallback_bounds_future_result(monkeypatch):
    from app.tools import amap_mcp_tools

    async def normal(value: str):
        return value

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(normal)])[0]
    future = MagicMock()
    future.result.return_value = "fallback-result"
    _force_running_loop_fallback(monkeypatch, amap_mcp_tools, future)
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.25),
    )

    assert wrapped._run(value="input") == "fallback-result"
    future.result.assert_called_once_with(timeout=0.25)


def test_running_loop_fallback_cancels_future_and_raises_stable_timeout(monkeypatch):
    from app.tools import amap_mcp_tools

    async def slow(value: str):
        return value

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(slow, name="slow-tool")])[0]
    future = MagicMock()
    future.result.side_effect = FutureTimeoutError
    _force_running_loop_fallback(monkeypatch, amap_mcp_tools, future)
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.25),
    )

    with pytest.raises(TimeoutError, match="MCP tool 'slow-tool' timed out"):
        wrapped._run(value="input")

    future.result.assert_called_once_with(timeout=0.25)
    future.cancel.assert_called_once_with()

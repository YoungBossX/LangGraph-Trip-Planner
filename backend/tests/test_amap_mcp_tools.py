import asyncio
import gc
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

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


def test_async_wrapper_rejects_result_returned_after_timeout(monkeypatch):
    from app.tools import amap_mcp_tools

    cancelled = threading.Event()

    async def return_after_cancellation(value: str):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            return "late result"

    wrapped = amap_mcp_tools.wrap_async_tools(
        [_structured_async_tool(return_after_cancellation, name="slow-tool")]
    )[0]
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


def test_sync_wrapper_returns_once_inside_real_running_loop(monkeypatch):
    from app.tools import amap_mcp_tools

    invocation_count = 0
    caller_thread_id = threading.get_ident()
    observed = {}

    async def normal(value: str):
        nonlocal invocation_count
        invocation_count += 1
        observed["tool_thread_id"] = threading.get_ident()
        observed["tool_thread_daemon"] = threading.current_thread().daemon
        observed["tool_loop_id"] = id(asyncio.get_running_loop())
        return f"result:{value}"

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(normal)])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=1),
    )

    async def main():
        observed["caller_loop_id"] = id(asyncio.get_running_loop())
        return wrapped._run(value="input")

    assert asyncio.run(main()) == "result:input"
    assert invocation_count == 1
    assert observed["tool_thread_id"] != caller_thread_id
    assert observed["tool_thread_daemon"] is False
    assert observed["tool_loop_id"] != observed["caller_loop_id"]


def test_worker_loop_cancels_spawned_tasks_before_closing(monkeypatch, caplog):
    from app.tools import amap_mcp_tools

    child_started = threading.Event()
    child_cancelled = threading.Event()
    child_completed = threading.Event()
    worker = {}

    async def child_task():
        child_started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        finally:
            child_completed.set()

    async def spawn_child(value: str):
        worker["thread"] = threading.current_thread()
        asyncio.create_task(child_task())
        await asyncio.sleep(0)
        return f"result:{value}"

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(spawn_child)])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=1),
    )

    async def main():
        return wrapped._run(value="input")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert asyncio.run(main()) == "result:input"
        assert child_started.wait(timeout=0.2)
        assert child_cancelled.wait(timeout=0.2)
        assert child_completed.wait(timeout=0.2)
        worker["thread"].join(timeout=0.2)
        gc.collect()

    assert not worker["thread"].is_alive()
    warning_messages = [str(warning.message) for warning in caught]
    assert not any("never awaited" in message for message in warning_messages)
    assert "Task was destroyed but it is pending" not in caplog.text


def test_provider_runtime_error_with_asyncio_run_text_propagates_once(monkeypatch):
    from app.tools import amap_mcp_tools

    invocation_count = 0
    message = "provider failed: asyncio.run() cannot be called from a running event loop"

    async def fail(value: str):
        nonlocal invocation_count
        invocation_count += 1
        raise RuntimeError(message)

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(fail)])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.05),
    )

    async def main():
        return wrapped._run(value="input")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(main())

    assert str(exc_info.value) == message
    assert invocation_count == 1


def test_slow_tool_times_out_and_finishes_cancellation_inside_real_running_loop(monkeypatch, caplog):
    from app.tools import amap_mcp_tools

    cancelled = threading.Event()
    completed = threading.Event()
    worker = {}

    async def slow(value: str):
        worker["task"] = asyncio.current_task()
        worker["thread"] = threading.current_thread()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            completed.set()

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(slow, name="slow-tool")])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.01),
    )

    async def main():
        return wrapped._run(value="input")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TimeoutError):
            asyncio.run(main())
        assert cancelled.wait(timeout=0.2)
        assert completed.wait(timeout=0.2)
        worker["thread"].join(timeout=0.2)
        gc.collect()

    assert worker["task"].cancelled()
    assert worker["thread"].daemon is False
    assert not worker["thread"].is_alive()
    warning_messages = [str(warning.message) for warning in caught]
    assert not any("never awaited" in message for message in warning_messages)
    assert "Task was destroyed but it is pending" not in caplog.text


def test_running_loop_timeout_rejects_result_returned_during_cleanup_grace(monkeypatch):
    from app.tools import amap_mcp_tools

    cancelled = threading.Event()
    completed = threading.Event()

    async def return_after_cancellation(value: str):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.02)
            return "late result"
        finally:
            completed.set()

    wrapped = amap_mcp_tools.wrap_async_tools(
        [_structured_async_tool(return_after_cancellation, name="slow-tool")]
    )[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.01),
    )

    async def main():
        return wrapped._run(value="input")

    with pytest.raises(TimeoutError, match="MCP tool 'slow-tool' timed out"):
        asyncio.run(main())

    assert cancelled.wait(timeout=0.2)
    assert completed.wait(timeout=0.2)


def test_running_loop_bridge_bounds_slow_cancellation_with_stable_timeout(monkeypatch):
    from app.tools import amap_mcp_tools

    started = threading.Event()
    cancelled = threading.Event()
    completed = threading.Event()
    worker = {}
    thread_errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args.exc_value))

    async def slow_to_cancel(value: str):
        started.set()
        worker["thread"] = threading.current_thread()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.4)
            return "late result"
        finally:
            completed.set()

    wrapped = amap_mcp_tools.wrap_async_tools([_structured_async_tool(slow_to_cancel, name="slow-tool")])[0]
    monkeypatch.setattr(
        amap_mcp_tools,
        "get_settings",
        lambda: SimpleNamespace(mcp_tool_timeout_seconds=0.1),
    )

    async def main():
        return wrapped._run(value="input")

    call_started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="MCP tool 'slow-tool' timed out"):
        asyncio.run(main())
    elapsed = time.monotonic() - call_started_at

    assert started.wait(timeout=0.2)
    assert elapsed < 0.35
    assert cancelled.wait(timeout=0.2)
    assert completed.wait(timeout=0.8)
    worker["thread"].join(timeout=0.2)
    assert not worker["thread"].is_alive()
    assert thread_errors == []

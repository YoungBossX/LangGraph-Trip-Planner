"""高德地图MCP工具 (LangChain MCP适配器版本)"""

import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import List, Optional

import nest_asyncio

try:
    from langchain_mcp_adapters.tools import load_mcp_tools
    MCP_ADAPTERS_AVAILABLE = True
except ImportError:
    MCP_ADAPTERS_AVAILABLE = False
    load_mcp_tools = None
from langchain_core.tools import BaseTool, StructuredTool

from ..config import get_settings

# 设置日志记录
logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_UV_CACHE_DIR = _BACKEND_DIR / ".uv-cache"
_DEFAULT_UV_TOOL_DIR = _BACKEND_DIR / ".uv-tools"
_SYNC_BRIDGE_GRACE_SECONDS = 0.1
_WORKER_LOOP_CLEANUP_GRACE_SECONDS = 0.1


async def _invoke_async_tool_with_timeout(tool, args, kwargs, timeout):
    task = asyncio.create_task(tool._arun(*args, **kwargs))
    done, _ = await asyncio.wait((task,), timeout=timeout)
    if task in done:
        return task.result()

    task.cancel()
    with suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=_SYNC_BRIDGE_GRACE_SECONDS)
    raise TimeoutError


def _run_async_tool_in_new_loop(tool, args, kwargs, timeout):
    return asyncio.run(_invoke_async_tool_with_timeout(tool, args, kwargs, timeout))


def _resolve_worker_outcome(outcome):
    _, succeeded, payload = outcome
    if succeeded:
        return payload
    raise payload


def _run_cleanup_awaitable(loop, awaitable, deadline):
    remaining = max(0.0, deadline - time.monotonic())
    if remaining == 0:
        with suppress(AttributeError):
            awaitable.close()
        with suppress(AttributeError):
            awaitable.cancel()
        return

    cleanup_future = asyncio.ensure_future(awaitable, loop=loop)
    callback_active = True

    def stop_loop(_):
        if callback_active:
            loop.stop()

    cleanup_future.add_done_callback(stop_loop)
    timeout_handle = loop.call_later(remaining, loop.stop)
    loop.run_forever()
    callback_active = False
    timeout_handle.cancel()
    cleanup_future.remove_done_callback(stop_loop)
    if cleanup_future.done():
        with suppress(BaseException):
            cleanup_future.result()
    else:
        cleanup_future.cancel()


def _cleanup_worker_loop(loop):
    deadline = time.monotonic() + _WORKER_LOOP_CLEANUP_GRACE_SECONDS
    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        _run_cleanup_awaitable(loop, asyncio.gather(*pending, return_exceptions=True), deadline)

    if time.monotonic() < deadline:
        _run_cleanup_awaitable(loop, loop.shutdown_asyncgens(), deadline)

    remaining = max(0.0, deadline - time.monotonic())
    if remaining > 0:
        if sys.version_info >= (3, 12):
            shutdown_executor = loop.shutdown_default_executor(timeout=remaining)
        else:
            shutdown_executor = loop.shutdown_default_executor()
        _run_cleanup_awaitable(loop, shutdown_executor, deadline)


def _run_async_tool_in_worker_thread(tool, args, kwargs, timeout, deadline):
    result_future = concurrent.futures.Future()
    worker_ready = threading.Event()
    cancellation_requested = threading.Event()
    worker = {}

    def publish(succeeded, payload):
        result_future.set_result((time.monotonic(), succeeded, payload))

    def run():
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(tool._arun(*args, **kwargs))
            worker["loop"] = loop
            worker["task"] = task
            worker_ready.set()
            if cancellation_requested.is_set():
                task.cancel()
            result = loop.run_until_complete(task)
        except BaseException as error:
            publish(False, error)
        else:
            publish(True, result)
        finally:
            worker_ready.set()
            if loop is not None:
                try:
                    _cleanup_worker_loop(loop)
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

    thread = threading.Thread(target=run, name=f"mcp-sync-{tool.name}", daemon=False)
    thread.start()

    remaining = max(0.0, deadline - time.monotonic())
    try:
        outcome = result_future.result(timeout=remaining)
    except concurrent.futures.TimeoutError:
        if result_future.done():
            outcome = result_future.result()
            if outcome[0] <= deadline:
                return _resolve_worker_outcome(outcome)
    else:
        if outcome[0] <= deadline:
            return _resolve_worker_outcome(outcome)

    cancellation_requested.set()
    cleanup_deadline = time.monotonic() + _SYNC_BRIDGE_GRACE_SECONDS
    worker_ready.wait(timeout=_SYNC_BRIDGE_GRACE_SECONDS)
    loop = worker.get("loop")
    task = worker.get("task")
    if loop is not None and task is not None and not task.done():
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(task.cancel)
    thread.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    raise TimeoutError(f"MCP tool '{tool.name}' timed out after {timeout} seconds")


def _build_amap_mcp_connection() -> dict:
    """Build the stdio connection used to launch the AMap MCP server."""
    settings = get_settings()
    uv_cache_dir = os.environ.get("UV_CACHE_DIR") or str(_DEFAULT_UV_CACHE_DIR)
    uv_tool_dir = os.environ.get("UV_TOOL_DIR") or str(_DEFAULT_UV_TOOL_DIR)

    return {
        "command": "uvx",
        "args": ["--python", sys.executable, "amap-mcp-server"],
        "transport": "stdio",
        "env": {
            "AMAP_MAPS_API_KEY": settings.amap_api_key,
            "UV_CACHE_DIR": uv_cache_dir,
            "UV_TOOL_DIR": uv_tool_dir,
            "UV_PYTHON_DOWNLOADS": "never",
        },
    }


def wrap_async_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """包装异步工具以支持同步调用

    某些 MCP 工具可能只实现了异步方法 (_arun)，
    但 LangGraph 工具节点需要同步调用。
    此函数检查工具是否有 _arun 方法但没有 _run 方法，
    并创建一个支持同步调用的包装器。
    """
    wrapped_tools = []

    for tool in tools:
        # 检查工具是否已经是 StructuredTool 且有 _arun 但没有 _run
        has_arun = hasattr(tool, '_arun') and callable(tool._arun)
        has_run = hasattr(tool, '_run') and callable(tool._run)

        # 检查是否是 StructuredTool 实例（即使有 _run 方法，也可能抛出 NotImplementedError）
        is_structured_tool = isinstance(tool, StructuredTool)

        # 需要包装的情况：
        # 1. 有 _arun 但没有 _run
        # 2. 是 StructuredTool 且有 _arun（因为 StructuredTool._run 会抛出 NotImplementedError）
        if (has_arun and not has_run) or (is_structured_tool and has_arun):
            logger.debug(f"包装异步工具: {tool.name} (类型: {type(tool).__name__})")

            # 创建一个新类，继承自原始工具类
            class SyncWrapper(tool.__class__):
                def _run(self, *args, **kwargs):
                    """同步运行方法，内部调用异步方法"""
                    started_at = time.monotonic()
                    timeout = get_settings().mcp_tool_timeout_seconds
                    deadline = started_at + timeout
                    # 确保 kwargs 中有 config 参数
                    if 'config' not in kwargs:
                        kwargs['config'] = None
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        remaining = max(0.0, deadline - time.monotonic())
                        return _run_async_tool_in_new_loop(self, args, kwargs, remaining)
                    return _run_async_tool_in_worker_thread(self, args, kwargs, timeout, deadline)

            # 创建包装器实例，复制所有属性
            wrapper = SyncWrapper(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema if hasattr(tool, 'args_schema') else None,
                return_direct=tool.return_direct if hasattr(tool, 'return_direct') else False,
                verbose=tool.verbose if hasattr(tool, 'verbose') else False,
                callbacks=tool.callbacks if hasattr(tool, 'callbacks') else None,
                tags=tool.tags if hasattr(tool, 'tags') else None,
                metadata=tool.metadata if hasattr(tool, 'metadata') else None,
                response_format=tool.response_format if hasattr(tool, 'response_format') else "content",
            )

            # 复制其他可能需要的属性
            for attr in ['func', 'coroutine']:
                if hasattr(tool, attr):
                    with suppress(AttributeError):
                        setattr(wrapper, attr, getattr(tool, attr))

            wrapped_tools.append(wrapper)
        else:
            # 工具已经支持同步调用，直接使用
            wrapped_tools.append(tool)

    return wrapped_tools


async def create_amap_mcp_tools() -> List[BaseTool]:
    """创建高德地图MCP工具列表"""
    settings = get_settings()

    # 验证必要的配置
    if not settings.amap_api_key:
        logger.error("AMAP_API_KEY 未配置")
        return []

    if not MCP_ADAPTERS_AVAILABLE:
        raise RuntimeError("langchain_mcp_adapters 未安装，无法加载高德 MCP 工具")

    try:

        # 创建连接配置
        connection = _build_amap_mcp_connection()

        logger.info("正在连接高德地图MCP服务器...")

        # 使用 load_mcp_tools 直接加载工具
        tools = await load_mcp_tools(
            session=None,
            connection=connection,
            server_name="amap",
            tool_name_prefix=False
        )

        logger.info(f"✅ 从MCP服务器加载了 {len(tools)} 个工具")

        # 为工具添加自定义描述，增强可读性
        tool_descriptions = {
            "maps_text_search": "搜索高德地图的POI（兴趣点）信息，如景点、餐厅、酒店等",
            "maps_weather": "查询指定城市的天气信息，包括温度、天气状况、风力等",
            "maps_geocode": "地址编码，将地址转换为经纬度坐标",
            "maps_reverse_geocode": "逆地址编码，将经纬度坐标转换为地址",
            "maps_route_planning": "路线规划，提供驾车、步行、公交等出行方式的路线规划"
        }

        for tool in tools:
            tool_name = tool.name.lower()
            for key, description in tool_descriptions.items():
                if key in tool_name:
                    tool.description = description
                    break

        # 包装异步工具以支持同步调用
        tools = wrap_async_tools(tools)
        logger.info(f"包装后工具数量: {len(tools)}")

        return tools

    except Exception as e:
        logger.error(f"❌ 加载MCP工具失败: {str(e)}", exc_info=True)
        # 返回空列表，调用方应处理空工具情况
        return []


def get_amap_mcp_tools() -> List[BaseTool]:
    """同步获取MCP工具"""
    try:
        # 应用 nest_asyncio 以允许在已有事件循环中运行
        nest_asyncio.apply()
        return asyncio.run(create_amap_mcp_tools())
    except Exception as e:
        logger.error(f"❌ 同步获取MCP工具失败: {str(e)}", exc_info=True)
        return []


def get_amap_essential_tools() -> List[BaseTool]:
    """获取主要的高德地图工具（过滤 text_search + weather）"""
    settings = get_settings()

    if not settings.amap_api_key:
        logger.error("AMAP_API_KEY 未配置")
        return []

    try:
        # 使用异步函数加载工具，然后过滤出主要工具
        async def load_and_filter():
            connection = _build_amap_mcp_connection()

            tools = await load_mcp_tools(
                session=None,
                connection=connection,
                server_name="amap",
                tool_name_prefix=False
            )

            # 过滤出主要工具
            essential_tool_names = {"maps_text_search", "maps_weather"}
            filtered_tools = []
            for tool in tools:
                tool_name = tool.name.lower()
                for essential_name in essential_tool_names:
                    if essential_name in tool_name:
                        # 添加描述
                        if "maps_text_search" in tool_name:
                            tool.description = "搜索高德地图的POI（兴趣点）信息，如景点、餐厅、酒店等"
                        elif "maps_weather" in tool_name:
                            tool.description = "查询指定城市的天气信息，包括温度、天气状况、风力等"
                        filtered_tools.append(tool)
                        break

            return filtered_tools

        nest_asyncio.apply()
        tools = asyncio.run(load_and_filter())
        logger.info(f"✅ 加载了 {len(tools)} 个主要高德地图工具")

        # 包装异步工具以支持同步调用
        tools = wrap_async_tools(tools)
        logger.info(f"包装后工具数量: {len(tools)}")

        return tools

    except Exception as e:
        logger.error(f"❌ 加载主要工具失败: {str(e)}", exc_info=True)
        return []


# 全局工具缓存
_cached_tools: Optional[List[BaseTool]] = None

def get_cached_amap_tools() -> List[BaseTool]:
    """获取缓存的高德地图工具（避免重复创建）"""
    global _cached_tools

    if _cached_tools is None:
        logger.info("首次加载高德地图工具，建立缓存...")
        tools = get_amap_mcp_tools()

        # 如果自动加载失败，使用主要工具备用
        if not tools:
            logger.warning("自动加载工具失败，尝试使用主要工具...")
            tools = get_amap_essential_tools()

        if not tools:
            raise RuntimeError("高德 MCP 工具加载失败，请检查 uvx/amap-mcp-server/AMAP_API_KEY")

        _cached_tools = tools

    return _cached_tools


def clear_tools_cache():
    """清空工具缓存（用于测试或重新加载）"""
    global _cached_tools
    _cached_tools = None
    logger.info("工具缓存已清空")

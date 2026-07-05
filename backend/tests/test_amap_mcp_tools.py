import sys
from pathlib import Path


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

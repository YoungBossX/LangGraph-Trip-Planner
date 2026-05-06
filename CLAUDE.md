# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 后端
cd backend
pip install -r requirements.txt
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# 前端
cd frontend
npm install
npm run dev           # Vite 开发服务器 → http://localhost:5173
npm run build         # 生产构建（含 vue-tsc 类型检查）

# 离线评测
cd backend
python evals/eval_runner.py                          # 全部用例
python evals/eval_runner.py --limit 5                # 前5个
python evals/eval_runner.py --gate                   # 开启门禁（失败返回非0）
python evals/eval_runner.py --baseline evals/reports/xxx.json  # 与基线对比
```

## 环境变量

后端通过 `pydantic-settings` 从 `backend/.env` 读取（会自动加载，无需手动 dotenv）。

**必需：**
- `AMAP_API_KEY` — 高德地图 Web 服务 API Key（MCP 工具连接需要）
- `LLM_API_KEY` — LLM API Key（LangChain 标准读取，会被当做 OpenAI key）
- `LLM_BASE_URL` — LLM API 地址
- `LLM_MODEL_ID` — 模型名称

**可选：**
- `UNSPLASH_ACCESS_KEY` / `UNSPLASH_SECRET_KEY` — 景点图片服务
- `LANGCHAIN_TRACING=true` + `LANGCHAIN_API_KEY` — LangSmith 追踪
- `CORS_ORIGINS` — 允许的跨域来源，逗号分隔

前端 `frontend/.env`：`VITE_AMAP_WEB_JS_KEY`（高德 JS API Key）、`VITE_API_BASE_URL`（后端地址）。

## 架构核心

### LangGraph 工作流（4 节点 + 错误恢复）

```
search_attractions → check_weather → find_hotels → plan_itinerary → END
       ↓                 ↓               ↓               ↓
       └─────────────────┴───────────────┴──────→ handle_error
```

**每个节点内部是 Agent（ReAct 模式）自主调用 MCP 工具**，节点函数只做三件事：构造 query → 调用 agent → 解析 JSON 输出。Agent 定义在 `backend/app/agents/agents.py`，每个 Agent 有专属 prompt 模板，规定了工具调用次数上限和输出格式。

**工具分配规则**（`trip_planner_graph.py:53-61`）：
- 景点 Agent：`maps_text_search` + `maps_geo`（搜索 + 补坐标）
- 天气 Agent：`maps_weather`
- 酒店 Agent：`maps_text_search` + `maps_geo`（搜索 + 补坐标）
- 规划 Agent：无工具（纯文本推理，根据结构化输入生成 JSON）

**错误恢复**（`_route_after_error`）：失败节点最多重试 2 次 → 有部分数据（景点或天气）则跳至 `plan_itinerary` → 全无数据则生成备用计划（`_create_fallback_plan`）。`last_failed_node` 用于区分"同一节点重试"和"不同节点失败"。

### MCP 工具加载链

`amap_mcp_tools.py` → `get_cached_amap_tools()` → 全局缓存 → 通过 `langchain_mcp_adapters.load_mcp_tools` 连接 `uvx amap-mcp-server`（stdio 传输）。

**同步包装**：MCP 工具只实现了 `_arun`，但 LangGraph 工具节点需要同步调用。`wrap_async_tools()` 用 `nest_asyncio` + `asyncio.run()` 桥接。

### Agent 输出 → Pydantic 解析链

Agent 输出文本 → `_extract_agent_output()` → `_extract_json()` → `json.loads()` → Pydantic 模型。

`_extract_json()` 支持：markdown 代码块提取 → 花括号/方括号平衡解析 → **截断 JSON 修复**（`_try_repair_truncated_json`：找到最后一个完整 `}`，补上 `]`）。这是因为 LLM `max_tokens` 截断时 JSON 不闭合。

### 前端数据流

`Home.vue` 提交表单 → `api.ts:generateTripPlan()` → `POST /api/trip/plan` → 返回 `TripPlan` → 存入 `sessionStorage` → 路由跳转 `Result.vue` → 从 `sessionStorage` 读取并渲染（地图 + 折叠面板 + 天气卡片）。

编辑模式修改的是前端本地数据，保存时写回 `sessionStorage`。导出（图片/PDF）通过 `html2canvas` 截图实现。

### 评测体系

`evals/eval_runner.py` 读取 `eval_cases.jsonl`（每行一个 JSON 用例），逐条执行工作流，评估：成功率、约束满足率（景点数/餐食完整性/雨天无户外）、坐标覆盖率、天数一致性、耗时。支持门禁阈值（`--gate`）和基线对比（`--baseline`）。

## 关键约定

- 所有服务/工具/工作流实例使用**模块级全局单例** + `get_xxx()` / `reset_xxx()` 模式
- Agent 内复用 Agent 缓存（`get_agent()` 按 `(agent_type, tool_names)` 组合缓存）
- 不要修改 `.env` 文件，不要提交密钥
- 模型新增字段时同步更新 `frontend/src/types/index.ts` 中的 TypeScript 类型

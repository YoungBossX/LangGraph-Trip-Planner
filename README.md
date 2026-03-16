# LangGraph 多 Agent 智能旅行助手 🌍✈️

基于 LangGraph 框架构建的多 Agent 智能旅行规划系统，集成高德地图 MCP 服务，通过"代码搜索 + Agent 智能筛选 + 代码地理编码"的混合架构，生成带有精确坐标和地图标记的个性化旅行计划。

## ✨ 功能特点

- 🤖 **多 Agent 协作**: 基于 LangGraph StateGraph 的 4 节点工作流（景点搜索 → 天气查询 → 酒店搜索 → 行程规划）
- 🗺️ **高德地图 MCP 集成**: 通过 MCP 协议接入 16 个高德地图工具，支持 POI 搜索、地理编码、天气查询、路线规划
- 📍 **精确地图标记**: 代码端批量地理编码补坐标，前端高德地图 JS API 展示景点标记和路线
- 🧠 **混合架构**: 确定性工作（搜索、编码）交给代码，创造性工作（筛选、规划）交给 AI Agent
- 🎨 **现代化前端**: Vue 3 + TypeScript + Ant Design Vue，响应式设计
- ⚡ **容错降级**: 每个节点独立容错，Agent 失败自动降级到代码兜底

## 🏗️ 技术架构

### 工作流节点

```
search_attractions → check_weather → find_hotels → plan_itinerary → END
        ↓ error           ↓ error         ↓ error        ↓ error
                        handle_error → END
```

| 节点 | 工作方式 | 说明 |
|------|---------|------|
| search_attractions | 代码多关键词搜索 → Agent 筛选/描述 → 代码 geo 补坐标 | 多偏好分别搜索，去重取 Top 8 |
| check_weather | Agent 调工具 → 解析，失败则代码兜底 | 高德天气 API 返回未来 4 天预报 |
| find_hotels | 代码搜索 → Agent 筛选/描述 → 代码 geo 补坐标 | 按住宿偏好搜索，取 Top 5 |
| plan_itinerary | Agent 综合所有数据生成完整 JSON 行程 | 唯一需要大量 LLM 推理的节点 |

### 技术栈

| 层级 | 技术 |
|------|------|
| 工作流编排 | LangGraph (StateGraph) |
| Agent 框架 | LangChain (`create_agent`) |
| MCP 工具 | amap-mcp-server (高德地图 16 个工具) |
| LLM | 兼容 OpenAI API (DeepSeek, Qwen 等) |
| 后端 API | FastAPI |
| 前端框架 | Vue 3 + TypeScript + Vite |
| UI 组件库 | Ant Design Vue |
| 地图渲染 | 高德地图 JavaScript API |

## 📁 项目结构

```
LangGraph-Trip-Planner/
├── backend/
│   ├── app/
│   │   ├── workflows/              # LangGraph 工作流
│   │   │   ├── trip_planner_graph.py   # 核心工作流（4 节点 + 错误处理）
│   │   │   └── trip_planner_state.py   # TypedDict 状态定义
│   │   ├── agents/
│   │   │   └── agents.py           # 4 个 Agent 定义和 Prompt
│   │   ├── api/
│   │   │   ├── main.py             # FastAPI 应用入口
│   │   │   └── routes/
│   │   │       ├── trip.py         # POST /api/trip/plan
│   │   │       ├── poi.py          # POI 搜索接口
│   │   │       └── map.py          # 地图相关接口
│   │   ├── services/
│   │   │   ├── llm_service.py      # ChatOpenAI 单例
│   │   │   ├── amap_service.py     # 高德地图直接调用
│   │   │   └── unsplash_service.py # 图片服务（可选）
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic 数据模型
│   │   ├── tools/
│   │   │   └── amap_mcp_tools.py   # MCP 工具加载和缓存
│   │   └── config.py               # 配置管理
│   └── run.py                      # 启动脚本
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── Home.vue            # 旅行表单页
│   │   │   └── Result.vue          # 结果展示页（地图 + 行程）
│   │   ├── services/
│   │   │   └── api.ts              # Axios HTTP 客户端
│   │   ├── types/
│   │   │   └── index.ts            # TypeScript 类型定义
│   │   ├── App.vue
│   │   └── main.ts                 # 路由定义
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- npm 或 yarn
- 高德地图 API Key（Web 服务 + JS API 安全密钥）
- LLM API Key（支持 OpenAI API 格式的服务商）

### 后端安装

```bash
cd backend

# 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
pip install fastapi uvicorn langchain langchain-openai langgraph langchain-mcp-adapters pydantic pydantic-settings python-dotenv httpx

# 配置环境变量
# 创建 .env 文件，填入以下内容：
```

**`.env` 文件模板：**

```dotenv
# 高德地图
AMAP_API_KEY=你的高德Web服务Key

# LLM 配置
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL_ID=deepseek-ai/DeepSeek-V3.2

# 可选：Unsplash 图片
UNSPLASH_ACCESS_KEY=
UNSPLASH_SECRET_KEY=

# Agent 配置（可选）
AGENT_MAX_ITERATIONS=5
AGENT_TEMPERATURE=0.7
AGENT_TIMEOUT=300.0
```

```bash
# 启动后端
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 在 src/views/Result.vue 中配置高德 JS API Key
# 搜索 AMapLoader.load 找到 key 字段

# 启动开发服务器
npm run dev
```

访问 `http://localhost:5173` 开始使用。

## 📝 使用说明

1. 填写目的地城市、旅行日期（只能选今天及以后）
2. 选择交通方式、住宿偏好、旅行偏好标签
3. 可选填写额外要求
4. 点击"开始规划我的旅行"
5. 等待 30-60 秒（取决于 LLM 响应速度）
6. 查看结果：每日行程、景点地图标记、路线连接、天气预报、预算明细

## 🔧 核心实现

### 景点搜索（代码 + Agent 混合）

```python
# 1. 代码多关键词搜索（快速、可靠）
for kw in ["杭州旅游景点", "杭州历史古迹", "杭州公园风景区"]:
    pois += search_tool.invoke({"keywords": kw, "city": "杭州", "citylimit": "true"})

# 2. Agent 智能筛选和描述（需要理解力）
agent_result = attraction_agent.invoke("从以下 POI 中选出最适合旅游的 8 个...")

# 3. 代码批量地理编码（快速、可靠）
for attr in attractions:
    geo_result = geo_tool.invoke({"address": f"杭州{attr.address}", "city": "杭州"})
    attr.location = extract_location(geo_result)
```

### 高德 MCP 工具

本项目使用的高德地图 MCP 工具：

| 工具 | 用途 |
|------|------|
| `maps_text_search` | POI 关键词搜索（景点、酒店） |
| `maps_geo` | 地址转经纬度坐标 |
| `maps_weather` | 城市天气预报查询 |
| `maps_search_detail` | POI 详情查询 |
| `maps_regeocode` | 经纬度转地址 |
| `maps_around_search` | 周边搜索 |
| 路线规划系列 | 步行/驾车/公交/骑行路线 |
| `maps_distance` | 距离测量 |

## 📄 API 文档

启动后端后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/trip/plan` | POST | 生成旅行计划 |
| `/api/map/poi` | GET | 搜索 POI |
| `/api/map/weather` | GET | 查询天气 |
| `/api/map/route` | POST | 规划路线 |

## ⚠️ 已知限制

- 高德天气 API 只返回未来 4 天预报，无法查询历史天气
- `maps_text_search` 返回的 POI 不含经纬度，需要额外调用 `maps_geo` 补充
- LLM 响应速度取决于 API 提供商，硅基流动高峰期可能超时
- 行程规划质量取决于 LLM 能力，建议使用 DeepSeek-V3.2 或以上模型

## 🤝 贡献

欢迎提交 Pull Request 或 Issue！

## 📜 开源协议

CC BY-NC-SA 4.0

## 🙏 致谢

- [Datawhale hello-agents](https://github.com/datawhalechina/hello-agents) — 原始项目
- [LangGraph](https://github.com/langchain-ai/langgraph) — 多 Agent 工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用框架
- [高德地图开放平台](https://lbs.amap.com/) — 地图服务
- [amap-mcp-server](https://github.com/amap-mcp/amap-mcp-server) — 高德地图 MCP 服务器

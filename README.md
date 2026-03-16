# 基于 LangGraph 框架构建的多 Agent 智能旅行规划系统 🌍✈️

基于LangGraph框架构建的多 Agent 智能旅行助手，集成高德地图MCP服务，提供个性化的旅行计划生成。

## ✨ 功能特点

- 🤖 **AI驱动的旅行规划**: 基于LangGraph框架的多智能体工作流,智能生成详细的多日旅程
- 🗺️ **高德地图集成**: 通过MCP协议接入高德地图服务,支持景点搜索、路线规划、天气查询
- 🧠 **智能工具调用**: Agent自动调用高德地图MCP工具,获取实时POI、路线和天气信息
- 🎨 **现代化前端**: Vue3 + TypeScript + Vite,响应式设计,流畅的用户体验
- 📱 **完整功能**: 包含住宿、交通、餐饮和景点游览时间推荐

## 🏗️ 技术栈

### 后端

- **框架**: LangGraph (基于StateGraph的多智能体工作流)
- **API**: FastAPI
- **MCP工具**: amap-mcp-server (高德地图)
- **LLM**: 支持多种LLM提供商(OpenAI, DeepSeek等)

### 前端

- **框架**: Vue 3 + TypeScript
- **构建工具**: Vite
- **UI组件库**: Ant Design Vue
- **地图服务**: 高德地图 JavaScript API
- **HTTP客户端**: Axios

## 📁 项目结构

```
LangGraph-Trip-Planner/
├── backend/
│   ├── app/
│   │   ├── workflows/
│   │   │   ├── trip_planner_graph.py      # LangGraph 核心工作流
│   │   │   └── trip_planner_state.py      # 状态定义与初始状态
│   │   ├── agents/
│   │   │   └── agents.py                  # 景点 / 天气 / 酒店 / 行程 Agent 定义
│   │   ├── api/
│   │   │   ├── main.py                    # FastAPI 应用入口
│   │   │   └── routes/
│   │   │       ├── trip.py                # 旅行规划接口
│   │   │       ├── poi.py                 # POI 与图片接口
│   │   │       └── map.py                 # 地图 / 路线 / 天气接口
│   │   ├── services/
│   │   │   ├── llm_service.py             # LLM 单例封装
│   │   │   ├── amap_service.py            # 高德工具服务封装
│   │   │   └── unsplash_service.py        # 景点图片服务
│   │   ├── models/
│   │   │   └── schemas.py                 # Pydantic 数据模型
│   │   ├── tools/
│   │   │   └── amap_mcp_tools.py          # 高德 MCP 工具加载与缓存
│   │   └── config.py                      # 全局配置
│   ├── env.example                        # 后端环境变量模板
│   ├── requirements.txt                   # 后端依赖（如仓库中已提交）
│   └── run.py                             # 后端启动脚本
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── Home.vue                   # 表单输入页
│   │   │   └── Result.vue                 # 行程结果页（地图 / 导出 / 编辑）
│   │   ├── services/
│   │   │   └── api.ts                     # API 请求封装
│   │   ├── types/
│   │   │   └── index.ts                   # 前端类型定义
│   │   ├── App.vue
│   │   └── main.ts                        # Vue 应用入口与路由
│   ├── env.example                        # 前端环境变量模板
│   ├── index.html
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- 高德地图API密钥 (Web服务API和Web端(JS API))
- LLM API密钥 (OpenAI/DeepSeek等)

### 后端安装

1. 进入后端目录

```bash
cd backend
```

2. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
# Mac: source venv/bin/activate 
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量

```bash
cp .env.example .env
# 编辑.env文件,填入你的API密钥
```

5. 启动后端服务

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

```

### 前端安装

1. 进入前端目录

```bash
cd frontend
```

2. 安装依赖

```bash
npm install
```

3. 配置环境变量

```bash
# 创建 .env 文件, 填入高德地图 Web API Key 和 Web 端 JS API Key
cp .env.example .env
```

4. 启动开发服务器

```bash
npm run dev
```

5. 打开浏览器访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息:
   - 目的地城市
   - 旅行日期和天数
   - 交通方式偏好
   - 住宿偏好
   - 旅行风格标签

2. 点击"生成旅行计划"按钮

3. 系统将:
   - 调用 LangGraph 工作流生成初步计划
   - Agent自动调用高德地图 MCP 工具搜索景点
   - Agent获取天气信息和路线规划
   - 整合所有信息生成完整行程

4. 查看结果:
   - 每日详细行程
   - 景点信息与地图标记
   - 交通路线规划
   - 天气预报
   - 餐饮推荐

## 🔧 核心实现

### LangGraph工作流集成

```python
from langgraph.graph import StateGraph, END
from app.workflows.trip_planner_graph import TripPlannerWorkflow
from app.models.schemas import TripRequest

# 创建旅行规划工作流
workflow = TripPlannerWorkflow()

# 创建旅行请求
request = TripRequest(
    city="北京",
    travel_days=3,
    transportation="地铁",
    accommodation="经济型酒店",
    preferences=["历史文化", "公园"],
    start_date="2024-10-01",
    end_date="2024-10-03",
    free_text_input="希望行程轻松一些"
)

# 执行工作流生成旅行计划
trip_plan = workflow.plan_trip(request)
print(f"生成 {len(trip_plan.days)} 天行程计划")
```

### MCP工具调用

工作流中的智能体可以自动调用以下高德地图MCP工具:

- `maps_text_search`: 搜索景点POI
- `maps_weather`: 查询天气
- `maps_direction_walking_by_address`: 步行路线规划
- `maps_direction_driving_by_address`: 驾车路线规划
- `maps_direction_transit_integrated_by_address`: 公共交通路线规划

## 📄 API文档

启动后端服务后,访问 `http://localhost:8000/docs` 查看完整的API文档。

主要端点:

- `POST /api/trip/plan` - 生成旅行计划
- `GET /api/map/poi` - 搜索POI
- `GET /api/map/weather` - 查询天气
- `POST /api/map/route` - 规划路线

## 🤝 贡献指南

欢迎提交Pull Request或Issue!

## 📜 开源协议

CC BY-NC-SA 4.0

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 多智能体工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- [高德地图开放平台](https://lbs.amap.com/) - 地图服务
- [amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) - 高德地图MCP服务器

---

**多Agent智能旅行助手** - 让旅行计划变得简单而智能 🌈

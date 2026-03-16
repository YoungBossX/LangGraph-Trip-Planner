# 🌍✈️ LangGraph-Trip-Planner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)](https://github.com/langchain-ai/langgraph)
[![Vue3](https://img.shields.io/badge/Vue.js-3.0-4FC08D.svg)](https://vuejs.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> **基于 LangGraph 框架构建的多 Agent 智能旅行规划系统**
> 
> 本项目采用“代码搜索 + Agent 智能筛选 + 代码地理编码”的**混合架构 (Hybrid Architecture)**，深度集成高德地图 MCP 服务。通过灵活的状态图（StateGraph）流转，系统能够自主生成带有精确坐标、丰富地图标记以及合理路线规划的个性化旅行方案。

---

## ✨ 核心特性

### 🤖 智能与 Agent 架构
- **多 Agent 协同工作流**: 基于 LangGraph `StateGraph` 构建的 4 节点流水线（景点检索 → 天气获取 → 酒店筛选 → 行程统筹），状态管理清晰，易于扩展。
- **LangChain 深度集成**: 底层利用 LangChain 的 `create_agent` 与各类 Tools 绑定，支持通过强大的 Prompt 工程与工具调用（Tool Calling）完成复杂推理。
- **混合决策机制**: 将确定性任务（如高并发的 POI 搜索、地理编码）交由底层代码处理，将创造性与评估任务（如偏好匹配、行程规划）交由 LLM 推理，最大化系统吞吐量与生成质量。
- **高鲁棒性与容错 (Fallback)**: 每个 Graph 节点均设计了独立的异常捕获机制，Agent 推理失败时可无缝降级至代码兜底逻辑。

### 💻 工程与交互实现
- **高德地图 MCP 无缝接入**: 通过标准 MCP 协议调用 16 个高德工具（涵盖 POI 检索、路线规划、地理编码等），打通虚拟规划与真实地理数据的壁垒。
- **精确地图渲染**: 前端基于 Vue 3 + 高德 JS API，将后端计算的经纬度精确映射为可视化标记与连线。
- **现代化响应式 UI**: 采用 Ant Design Vue 构建，提供极佳的用户输入体验与清晰的行程流展示。

---

## 🏗️ 系统架构

### 核心状态流转 (StateGraph)

基于 LangGraph 的核心工作流如下所示。系统在 `State` 中维护全局上下文（包括用户偏好、已选景点、天气状态等），并在各个节点间高效传递。

```mermaid
graph TD
    Start((开始规划)) --> A[🔍 search_attractions]
    A --> B[⛅ check_weather]
    B --> C[🏨 find_hotels]
    C --> D[🗺️ plan_itinerary]
    D --> End((输出完整 JSON))

    A -.->|Error/Fallback| F[🛠️ handle_error]
    B -.->|Error/Fallback| F
    C -.->|Error/Fallback| F
    D -.->|Error/Fallback| F
    F -.-> End

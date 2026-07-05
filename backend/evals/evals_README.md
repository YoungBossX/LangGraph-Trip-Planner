# 离线评测说明

## 目录结构

```
evals/
├── eval_cases.jsonl    # 评测用例（20条，覆盖15个城市）
├── eval_runner.py      # 评测脚本
├── reports/            # 生成的报告（自动创建）
└── evals_README.md     # 本文件
```

## 快速开始

```bash
cd backend

# 运行全部20个用例
python evals/eval_runner.py

# 只跑前3个（快速验证）
python evals/eval_runner.py --limit 3

# 开启门禁检查
python evals/eval_runner.py --gate

# 与基线报告对比
python evals/eval_runner.py --baseline evals/reports/eval_report_20260316_120000.json
```

## 评测指标详解

### 1. 成功率（success_rate）
- **含义**: 工作流正常返回 TripPlan 的比例（系统不含 fallback 假数据机制，失败即报错）
- **计算**: 成功用例数 / 总用例数
- **门禁阈值**: ≥ 90%
- **分析**: 低于阈值说明 Agent 调用工具不稳定（超时、MCP 连接失败等）

### 2. 约束满足率（constraint_satisfaction_rate）
- **含义**: 满足**所有**约束条件的用例比例
- **检查项**:
  - 每天景点数在 [min, max] 范围内
  - 每天必须包含 breakfast/lunch/dinner 三餐
  - 雨天不安排高暴露户外景点（如爬山、徒步、海滨）
- **门禁阈值**: ≥ 80%
- **分析**: 低说明 Planner Agent 的 Prompt 需要加强约束描述

### 3. 天数一致率（days_match_rate）
- **含义**: 生成的行程天数与用户请求天数一致的比例
- **计算**: 天数匹配用例数 / 总用例数
- **分析**: 不一致说明 Planner Agent 忽略了 travel_days 参数

### 4. 坐标覆盖率（location_coverage_rate）
- **含义**: 景点+酒店中有经纬度坐标的比例
- **计算**: (有坐标的景点数 + 有坐标的酒店数) / (总景点数 + 总酒店数)
- **门禁阈值**: ≥ 70%
- **分析**: 低说明 maps_geo 地理编码失败率高，检查地址拼接逻辑

### 5. 数据真实率（data_authenticity_rate）★新增
- **含义**: 所有数据（景点名/地址/坐标/天气日期）通过真实性检测的用例比例
- **检查项**:
  - 景点名不包含 `{city}景点{N}`、`著名景点` 等虚假模板
  - 地址不包含 `某区某路` 等占位符，不限于纯城市名
  - 所有景点坐标不全部相同（检测坐标重复）
  - 坐标不指向与目的地不符的城市（如所有景点坐标在北京但目的地是杭州）
  - 天气日期与行程日期匹配
  - 描述不包含 `历史悠久，值得一游` 等模板化表述
- **门禁阈值**: ≥ 80%
- **分析**: 低说明 LLM 在编造数据而非使用工具返回的真实数据

### 6. 平均耗时（avg_latency_ms）
- **含义**: 成功用例的平均总耗时（毫秒）
- **门禁阈值**: ≤ 120000ms (2分钟)
- **分析**: 主要耗时在 LLM 调用，可通过换更快的模型或减少 Agent 调用次数优化

## 用例格式

```json
{
  "id": "case_0001",
  "input": {
    "city": "杭州",
    "start_date": "2026-04-10",
    "end_date": "2026-04-11",
    "travel_days": 2,
    "transportation": "公共交通",
    "accommodation": "经济型酒店",
    "preferences": ["历史文化", "美食"],
    "free_text_input": "希望每天节奏松一点"
  },
  "constraints": {
    "min_attractions_per_day": 2,
    "max_attractions_per_day": 3,
    "required_meal_types": ["breakfast", "lunch", "dinner"],
    "avoid_outdoor_on_rain": true
  }
}
```

## 门禁参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min-success-rate` | 0.90 | 成功率下限 |
| `--min-constraint-rate` | 0.80 | 约束满足率下限 |
| `--min-location-rate` | 0.70 | 坐标覆盖率下限 |
| `--min-authenticity-rate` | 0.80 | 数据真实率下限 |
| `--max-avg-latency-ms` | 120000 | 平均耗时上限(ms) |

## 报告输出

每次评测生成两个文件:
- `evals/reports/eval_report_YYYYMMDD_HHMMSS.json` — 机器可读的完整报告
- `evals/reports/eval_report_YYYYMMDD_HHMMSS.md` — 人可读的 Markdown 报告

## 自定义用例

可以用 LLM 批量生成用例，Prompt 模板:

```text
你是旅行规划离线评测数据生成器。请直接输出 JSONL（每行一个 JSON），不要 Markdown。

生成 50 条中国城市短途旅行评测用例:
- 覆盖 1-5 天行程
- 城市分布均衡（一线/新一线/旅游城市）
- preferences 从 ["历史文化","美食","自然风光","购物","艺术","休闲"] 选 2-3 个
- transportation 从 ["公共交通","自驾","步行","混合"] 选
- accommodation 从 ["经济型酒店","舒适型酒店","豪华酒店","民宿"] 选
- constraints 固定为 min_attractions_per_day=2, max_attractions_per_day=3
```

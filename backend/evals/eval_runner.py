"""
离线评测脚本 - 适配 LangGraph-Trip-Planner

评测维度:
1. 成功率 - 工作流是否正常返回结果
2. 约束满足率 - 景点数/餐食/雨天规则是否达标
3. 坐标覆盖率 - 景点和酒店是否有经纬度
4. 天数一致性 - 生成天数是否与请求一致
5. 耗时统计 - 各阶段耗时

用法:
    cd backend
    python evals/eval_runner.py                           # 运行全部用例
    python evals/eval_runner.py --limit 5                 # 只跑前5个
    python evals/eval_runner.py --gate                    # 开启门禁
    python evals/eval_runner.py --baseline evals/reports/xxx.json  # 与基线对比
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# 把 backend/ 加入 sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("AMAP_MAPS_API_KEY") and os.environ.get("AMAP_API_KEY"):
    os.environ["AMAP_MAPS_API_KEY"] = os.environ["AMAP_API_KEY"]

from app.models.schemas import TripRequest
from app.workflows.trip_planner_graph import get_trip_planner_workflow, reset_workflow

# ---------- 雨天/户外判定词 ----------
RAIN_TERMS = ("雨", "雷阵雨", "暴雨", "雨夹雪", "雪", "storm", "shower", "rain")
OUTDOOR_TERMS = (
    "长城", "爬山", "登山", "徒步", "古道", "山", "峰", "岭",
    "森林公园", "湿地", "观景台", "露营", "漂流", "海滨", "沙滩", "海岛", "hiking",
)


@dataclass
class CaseResult:
    """单个用例的评测结果"""
    case_id: str
    city: str
    travel_days: int
    status: str
    error: str = ""
    constraint_passed: bool = False
    violations: List[str] = field(default_factory=list)

    # 天数一致性
    expected_days: int = 0
    actual_days: int = 0
    days_match: bool = False

    # 坐标覆盖
    total_attractions: int = 0
    attractions_with_location: int = 0
    total_hotels: int = 0
    hotels_with_location: int = 0
    location_coverage: float = 0.0

    # 耗时 (ms)
    total_ms: int = 0


# ---------- 工具函数 ----------

def _read_cases(path: Path) -> List[Dict[str, Any]]:
    cases = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {idx} 行 JSON 格式错误: {exc}") from exc
    return cases


def _is_rainy(day_weather: str, night_weather: str) -> bool:
    text = f"{day_weather} {night_weather}".lower()
    return any(term in text for term in RAIN_TERMS)


def _is_outdoor(attr: Any) -> bool:
    name = getattr(attr, "name", "") or ""
    category = getattr(attr, "category", "") or ""
    desc = getattr(attr, "description", "") or ""
    text = f"{name} {category} {desc}".lower()
    return any(term in text for term in OUTDOOR_TERMS)


def _evaluate_constraints(plan: Any, constraints: Dict[str, Any]) -> List[str]:
    """检查行程是否满足约束"""
    violations = []
    if plan is None:
        return ["trip_plan is null"]

    min_cnt = constraints.get("min_attractions_per_day", 2)
    max_cnt = constraints.get("max_attractions_per_day", 3)
    required_meals = [m.lower() for m in constraints.get("required_meal_types", ["breakfast", "lunch", "dinner"])]
    avoid_rain = constraints.get("avoid_outdoor_on_rain", True)

    # 天气按日期索引
    weather_by_date = {}
    for w in (getattr(plan, "weather_info", []) or []):
        d = getattr(w, "date", "")
        if d:
            weather_by_date[d] = w

    for day in (getattr(plan, "days", []) or []):
        day_date = getattr(day, "date", "")
        day_idx = getattr(day, "day_index", "?")
        label = f"第{day_idx}天({day_date})"

        # 1. 景点数量
        attractions = getattr(day, "attractions", []) or []
        count = len(attractions)
        if count < min_cnt or count > max_cnt:
            violations.append(f"{label} 景点数={count}, 要求[{min_cnt},{max_cnt}]")

        # 2. 餐食完整性
        meal_types = {(getattr(m, "type", "") or "").lower() for m in (getattr(day, "meals", []) or [])}
        for mt in required_meals:
            if mt not in meal_types:
                violations.append(f"{label} 缺少{mt}")

        # 3. 雨天户外
        weather = weather_by_date.get(day_date)
        if avoid_rain and weather:
            if _is_rainy(getattr(weather, "day_weather", ""), getattr(weather, "night_weather", "")):
                bad = [getattr(a, "name", "") for a in attractions if _is_outdoor(a)]
                if bad:
                    violations.append(f"{label} 雨天安排了户外: {', '.join(bad[:2])}")

    return violations


def _evaluate_location_coverage(plan: Any) -> Dict[str, Any]:
    """统计坐标覆盖率"""
    total_attr = 0
    attr_with_loc = 0
    total_hotel = 0
    hotel_with_loc = 0

    for day in (getattr(plan, "days", []) or []):
        for a in (getattr(day, "attractions", []) or []):
            total_attr += 1
            if getattr(a, "location", None) is not None:
                attr_with_loc += 1
        hotel = getattr(day, "hotel", None)
        if hotel and getattr(hotel, "name", ""):
            total_hotel += 1
            if getattr(hotel, "location", None) is not None:
                hotel_with_loc += 1

    total = total_attr + total_hotel
    with_loc = attr_with_loc + hotel_with_loc
    coverage = with_loc / total if total > 0 else 0.0

    return {
        "total_attractions": total_attr,
        "attractions_with_location": attr_with_loc,
        "total_hotels": total_hotel,
        "hotels_with_location": hotel_with_loc,
        "location_coverage": round(coverage, 4),
    }


def _safe_mean(values: List[int]) -> Optional[float]:
    return round(float(mean(values)), 2) if values else None


def _safe_pct(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------- 运行评测 ----------

def run_eval(cases: List[Dict], reset_each: bool = False) -> List[CaseResult]:
    """逐个运行用例，收集结果"""
    results = []

    if not reset_each:
        reset_workflow()
        workflow = get_trip_planner_workflow()
    else:
        workflow = None

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{idx:04d}")
        inp = case.get("input", {})
        constraints = case.get("constraints", {})
        city = inp.get("city", "?")
        travel_days = inp.get("travel_days", 0)

        print(f"[{idx}/{len(cases)}] {case_id} ({city}, {travel_days}天) ...", end=" ", flush=True)

        if reset_each:
            reset_workflow()
            workflow = get_trip_planner_workflow()

        # 构造请求
        try:
            request = TripRequest(**inp)
        except Exception as exc:
            print(f"输入错误: {exc}")
            results.append(CaseResult(
                case_id=case_id, city=city, travel_days=travel_days,
                status="input_error", error=str(exc),
                violations=[f"输入错误: {exc}"],
                expected_days=travel_days,
            ))
            continue

        # 执行工作流
        plan = None
        error_msg = ""
        status = "success"
        start_time = time.perf_counter()

        try:
            plan = workflow.plan_trip(request)
        except Exception as exc:
            status = "runtime_error"
            error_msg = str(exc)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # 评估约束
        violations = _evaluate_constraints(plan, constraints) if status == "success" else [f"运行错误: {error_msg}"]

        # 天数一致性
        actual_days = len(getattr(plan, "days", []) or []) if plan else 0
        days_match = actual_days == travel_days

        if not days_match and status == "success":
            violations.append(f"天数不一致: 期望{travel_days}天, 实际{actual_days}天")

        # 坐标覆盖
        loc_info = _evaluate_location_coverage(plan) if plan else {}

        result = CaseResult(
            case_id=case_id, city=city, travel_days=travel_days,
            status=status, error=error_msg,
            constraint_passed=(len(violations) == 0),
            violations=violations,
            expected_days=travel_days,
            actual_days=actual_days,
            days_match=days_match,
            total_attractions=loc_info.get("total_attractions", 0),
            attractions_with_location=loc_info.get("attractions_with_location", 0),
            total_hotels=loc_info.get("total_hotels", 0),
            hotels_with_location=loc_info.get("hotels_with_location", 0),
            location_coverage=loc_info.get("location_coverage", 0.0),
            total_ms=elapsed_ms,
        )
        results.append(result)

        status_icon = "✅" if result.constraint_passed else ("❌" if status != "success" else "⚠️")
        print(f"{status_icon} {elapsed_ms}ms, 景点{result.total_attractions}个, "
              f"坐标{_fmt_pct(result.location_coverage)}, 违规{len(violations)}条")

    return results


def _build_summary(results: List[CaseResult]) -> Dict[str, Any]:
    """汇总所有用例的统计指标"""
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = total - success
    constraint_passed = sum(1 for r in results if r.constraint_passed)
    days_matched = sum(1 for r in results if r.days_match)

    # 坐标覆盖
    total_entities = sum(r.total_attractions + r.total_hotels for r in results)
    entities_with_loc = sum(r.attractions_with_location + r.hotels_with_location for r in results)

    latencies = [r.total_ms for r in results if r.status == "success"]

    return {
        "total_cases": total,
        "success_cases": success,
        "failed_cases": failed,
        "constraint_passed_cases": constraint_passed,
        "days_matched_cases": days_matched,
        "success_rate": _safe_pct(success, total),
        "failure_rate": _safe_pct(failed, total),
        "constraint_satisfaction_rate": _safe_pct(constraint_passed, total),
        "days_match_rate": _safe_pct(days_matched, total),
        "location_coverage_rate": _safe_pct(entities_with_loc, total_entities),
        "avg_latency_ms": _safe_mean(latencies),
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


def _summary_cn(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "总用例数": s["total_cases"],
        "成功数": s["success_cases"],
        "失败数": s["failed_cases"],
        "约束通过数": s["constraint_passed_cases"],
        "天数一致数": s["days_matched_cases"],
        "成功率": _fmt_pct(s["success_rate"]),
        "失败率": _fmt_pct(s["failure_rate"]),
        "约束满足率": _fmt_pct(s["constraint_satisfaction_rate"]),
        "天数一致率": _fmt_pct(s["days_match_rate"]),
        "坐标覆盖率": _fmt_pct(s["location_coverage_rate"]),
        "平均耗时(ms)": s["avg_latency_ms"],
        "最短耗时(ms)": s["min_latency_ms"],
        "最长耗时(ms)": s["max_latency_ms"],
    }


def _make_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# 评测报告", "",
        f"- 生成时间: {report['generated_at']}",
        f"- 用例路径: {report['cases_path']}",
        f"- 总用例数: {s['total_cases']}", "",
        "## 核心指标", "",
        f"| 指标 | 值 |",
        f"|------|---:|",
        f"| 成功率 | {_fmt_pct(s['success_rate'])} |",
        f"| 约束满足率 | {_fmt_pct(s['constraint_satisfaction_rate'])} |",
        f"| 天数一致率 | {_fmt_pct(s['days_match_rate'])} |",
        f"| 坐标覆盖率 | {_fmt_pct(s['location_coverage_rate'])} |",
        f"| 平均耗时 | {s['avg_latency_ms']}ms |",
        f"| 最短耗时 | {s['min_latency_ms']}ms |",
        f"| 最长耗时 | {s['max_latency_ms']}ms |",
        "",
    ]

    # 基线对比
    if report.get("baseline_comparison"):
        lines += ["## 与基线对比", ""]
        for k, v in report["baseline_comparison"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # 用例详情
    lines += [
        "## 用例详情", "",
        "| case_id | 城市 | 天数 | 状态 | 约束 | 坐标率 | 耗时(ms) | 违规 |",
        "|---------|------|---:|------|------|-------:|--------:|------|",
    ]
    for r in report["results"]:
        status_cn = {"success": "✅", "runtime_error": "❌", "input_error": "⚠️"}.get(r["status"], r["status"])
        violations = "; ".join(r["violations"][:2]) or "-"
        lines.append(
            f"| {r['case_id']} | {r['city']} | {r['travel_days']} | {status_cn} | "
            f"{'✅' if r['constraint_passed'] else '❌'} | "
            f"{_fmt_pct(r['location_coverage'])} | {r['total_ms']} | {violations} |"
        )
    lines.append("")

    # 门禁
    gate = report.get("gate", {})
    if gate.get("enabled"):
        lines += [
            "## 门禁结果", "",
            f"- 是否通过: {'✅' if gate['passed'] else '❌'}",
        ]
        for reason in gate.get("reasons", []):
            lines.append(f"- ❌ {reason}")

    return "\n".join(lines)


def _compare_baseline(current: Dict, baseline_path: Path) -> Dict[str, str]:
    if not baseline_path.exists():
        return {"错误": f"基线文件不存在: {baseline_path}"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    bs = baseline.get("summary", {})
    out = {}
    keys = {
        "success_rate": "成功率", "constraint_satisfaction_rate": "约束满足率",
        "days_match_rate": "天数一致率", "location_coverage_rate": "坐标覆盖率",
        "avg_latency_ms": "平均耗时(ms)",
    }
    for key, label in keys.items():
        cur, old = current.get(key), bs.get(key)
        if cur is None or old is None:
            continue
        try:
            c, o = float(cur), float(old)
            delta = c - o
            out[label] = f"{c:.4f} (基线 {o:.4f}, 变化 {delta:+.4f})"
        except Exception:
            out[label] = f"{cur} (基线 {old})"
    return out


# ---------- 主入口 ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph 旅行规划离线评测")
    parser.add_argument("--cases", default="evals/eval_cases.jsonl", help="用例文件路径")
    parser.add_argument("--output", default="", help="报告输出路径")
    parser.add_argument("--baseline", default="", help="基线报告路径（用于对比）")
    parser.add_argument("--limit", type=int, default=0, help="只运行前 N 个用例")
    parser.add_argument("--gate", action="store_true", help="开启门禁检查")
    parser.add_argument("--reset-each", action="store_true", help="每个用例重建工作流")

    # 门禁阈值
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--min-constraint-rate", type=float, default=0.80)
    parser.add_argument("--min-location-rate", type=float, default=0.70)
    parser.add_argument("--max-avg-latency-ms", type=float, default=120000)

    args = parser.parse_args()

    # 解析路径
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = (BACKEND_DIR / cases_path).resolve()

    report_path = Path(args.output) if args.output else Path(
        f"evals/reports/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取用例
    cases = _read_cases(cases_path)
    if args.limit > 0:
        cases = cases[:args.limit]
    if not cases:
        print("没有找到评测用例")
        return 1

    print(f"\n{'='*60}")
    print(f"开始评测: {len(cases)} 个用例")
    print(f"{'='*60}\n")

    # 运行
    results = run_eval(cases, reset_each=args.reset_each)

    # 汇总
    summary = _build_summary(results)

    # 门禁
    gate_reasons = []
    if summary["success_rate"] < args.min_success_rate:
        gate_reasons.append(f"成功率 {_fmt_pct(summary['success_rate'])} < {_fmt_pct(args.min_success_rate)}")
    if summary["constraint_satisfaction_rate"] < args.min_constraint_rate:
        gate_reasons.append(f"约束满足率 {_fmt_pct(summary['constraint_satisfaction_rate'])} < {_fmt_pct(args.min_constraint_rate)}")
    if summary["location_coverage_rate"] < args.min_location_rate:
        gate_reasons.append(f"坐标覆盖率 {_fmt_pct(summary['location_coverage_rate'])} < {_fmt_pct(args.min_location_rate)}")
    if args.max_avg_latency_ms > 0 and summary["avg_latency_ms"] is not None:
        if summary["avg_latency_ms"] > args.max_avg_latency_ms:
            gate_reasons.append(f"平均耗时 {summary['avg_latency_ms']}ms > {args.max_avg_latency_ms}ms")

    gate = {
        "enabled": args.gate,
        "passed": len(gate_reasons) == 0,
        "reasons": gate_reasons,
    }

    # 生成报告
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases_path": str(cases_path),
        "summary": summary,
        "summary_cn": _summary_cn(summary),
        "gate": gate,
        "results": [asdict(r) for r in results],
    }

    if args.baseline:
        bl_path = Path(args.baseline)
        if not bl_path.is_absolute():
            bl_path = (BACKEND_DIR / bl_path).resolve()
        report["baseline_comparison"] = _compare_baseline(summary, bl_path)

    # 写入文件
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = report_path.with_suffix(".md")
    md_path.write_text(_make_markdown(report), encoding="utf-8")

    # 打印汇总
    print(f"\n{'='*60}")
    print("评测汇总")
    print(f"{'='*60}")
    for k, v in _summary_cn(summary).items():
        print(f"  {k}: {v}")
    print(f"\n报告 JSON: {report_path}")
    print(f"报告 MD  : {md_path}")

    # 门禁判定
    if args.gate:
        if gate_reasons:
            print(f"\n❌ 门禁未通过:")
            for r in gate_reasons:
                print(f"  - {r}")
            return 1
        else:
            print(f"\n✅ 门禁通过")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from evals.eval_runner import CaseResult, _build_evaluation_info, _build_summary


def _case(
    case_id: str,
    total_ms: int,
    *,
    status: str = "success",
    constraint_passed: bool = True,
    days_match: bool = True,
    auth_passed: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        city="Hangzhou",
        travel_days=2,
        status=status,
        constraint_passed=constraint_passed,
        expected_days=2,
        actual_days=2 if days_match else 1,
        days_match=days_match,
        total_attractions=4,
        attractions_with_location=4,
        total_hotels=2,
        hotels_with_location=2,
        location_coverage=1.0,
        total_ms=total_ms,
        auth_passed=auth_passed,
    )


def test_build_evaluation_info_includes_latency_quality_and_verdict():
    results = [
        _case("case_1", 1000),
        _case("case_2", 2000),
        _case("case_3", 3000, constraint_passed=False),
        _case("case_4", 4000, status="runtime_error", auth_passed=False),
    ]

    summary = _build_summary(results)
    info = _build_evaluation_info(summary, results)

    assert info["verdict"] == "needs_attention"
    assert info["latency"]["successful_cases"] == 3
    assert info["latency"]["avg_seconds"] == 2.0
    assert info["latency"]["p95_ms"] == 3000
    assert info["quality"]["success_rate"] == 0.75
    assert "提高成功率" in info["recommendations"]

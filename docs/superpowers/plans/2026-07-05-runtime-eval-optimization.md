# Runtime and Evaluation Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce end-to-end trip planning latency by parallelizing independent Agent work, and make offline evaluation reports expose clear runtime and quality conclusions.

**Architecture:** Keep the existing FastAPI/LangGraph/Vue architecture. Change only the backend workflow graph so weather and hotel collection run as sibling branches after attractions, then meet at a lightweight `context_ready` gate before planning. Extend the existing `eval_runner.py` rather than adding a second evaluator.

**Tech Stack:** Python 3.12, LangGraph, pytest, Ruff, FastAPI, Vue/Vite build verification.

---

### Task 1: Parallelize Independent Agent Nodes

**Files:**
- Modify: `backend/app/workflows/trip_planner_graph.py`
- Modify: `backend/app/workflows/trip_planner_state.py`
- Modify: `backend/app/api/routes/trip.py`
- Test: `backend/tests/test_workflow_recovery.py`

- [x] **Step 1: Add failing topology test**

Assert that `search_attractions` fans out to both `check_weather` and `find_hotels`, and that `check_weather` no longer serially points to `find_hotels`.

- [x] **Step 2: Add state reducers for parallel control fields**

Add a small reducer for `error` and `failed_node` so parallel branches can safely report failures and clear them during recovery.

- [x] **Step 3: Add `context_ready` gate**

Route both weather and hotel branches to a gate node. The gate proceeds to `plan_itinerary` only when weather and hotel data are present; otherwise it routes through existing error recovery.

- [x] **Step 4: Update SSE step labels**

Add a label for `context_ready` so streamed progress remains readable.

### Task 2: Add Evaluation Information Summary

**Files:**
- Modify: `backend/evals/eval_runner.py`
- Test: `backend/tests/test_eval_runner.py`

- [x] **Step 1: Add failing pure evaluator test**

Create synthetic `CaseResult` values and assert that the evaluator returns `verdict`, quality rates, average seconds, P95 latency, and recommendations.

- [x] **Step 2: Implement `evaluation_info`**

Add `_build_evaluation_info()` and nearest-rank P95 calculation. Include `evaluation_info` in JSON reports, Markdown reports, and console output.

### Task 3: Verification

**Files:**
- No additional production files.

- [x] **Step 1: Run focused tests**

Run the new workflow topology and evaluator tests.

- [x] **Step 2: Run backend test suite**

Run `..\.venv\Scripts\python -m pytest tests\ -v`.

- [x] **Step 3: Run backend lint**

Run `..\.venv\Scripts\python -m ruff check .`.

- [x] **Step 4: Run frontend build**

Run `npm run build`.

- [x] **Step 5: Check evaluator CLI**

Run `..\.venv\Scripts\python evals\eval_runner.py --help`.

# Phase A Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a reproducible, buildable baseline and fix the highest-risk runtime breaks without changing the product architecture.

**Architecture:** Keep the current FastAPI/LangGraph/Vue architecture. Add focused tests around the existing failure points, then make narrow changes to configuration, parsing/error semantics, SSE handling, frontend typing, and repository tracking rules.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, pytest, Ruff, Vue 3, TypeScript, Vite, Ant Design Vue.

---

### Task 1: Reproducible Backend Tooling

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/pyproject.toml`

- [x] **Step 1: Confirm current backend verification failure**

Run: `pytest tests/ -v`
Expected: FAIL because the active Python environment is missing LangGraph dependencies.

Run: `ruff check .`
Expected: FAIL because Ruff is not installed from the declared backend requirements.

- [x] **Step 2: Add missing declared tooling**

Add `ruff` to `backend/requirements.txt`.

- [x] **Step 3: Ensure pytest can import local app**

Add `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `backend/pyproject.toml`.

- [x] **Step 4: Verify**

Run after dependencies are available: `pytest tests/ -v`
Expected: PASS for existing parsing tests.

Run after dependencies are available: `ruff check .`
Expected: either PASS or actionable style failures from real code, not missing executable.

### Task 2: Backend Truthfulness and Recovery Semantics

**Files:**
- Modify: `backend/app/workflows/trip_planner_graph.py`
- Modify: `backend/tests/test_json_parsing.py`
- Create or modify: `backend/tests/test_workflow_recovery.py`

- [x] **Step 1: Write failing tests for empty parsed data**

Add tests asserting attraction, weather, and hotel nodes treat unparseable or empty Agent output as node failures rather than successful empty data.

- [x] **Step 2: Write failing tests for retry reset**

Add tests asserting `_handle_error()` resets retry count when a different node fails and clears `failed_node` only when skipping to planner.

- [x] **Step 3: Implement minimal backend changes**

Make each data-gathering node raise or return `error` when parsed output is empty. Preserve final planner parsing behavior.

- [x] **Step 4: Verify**

Run: `pytest tests/ -v`
Expected: PASS when dependencies are installed.

### Task 3: SSE Stream Semantics

**Files:**
- Modify: `backend/app/api/routes/trip.py`
- Modify: `frontend/src/services/api.ts`

- [x] **Step 1: Add or reason-test stream contract**

Backend should emit recoverable node failures as progress/recovery metadata, not terminal SSE `error`, unless the workflow ends without `trip_plan`.

- [x] **Step 2: Harden frontend event parsing**

Parse SSE by full event frames separated by blank lines so chunk boundaries and multi-line events do not corrupt state.

- [x] **Step 3: Verify frontend build**

Run: `npm run build`
Expected: build reaches Vite compile after TypeScript errors are fixed.

### Task 4: Frontend Type and API Contract Fixes

**Files:**
- Create: `frontend/src/vite-env.d.ts`
- Modify: `frontend/src/views/Home.vue`
- Modify: `frontend/src/views/Result.vue`
- Modify: `frontend/src/types/index.ts`

- [x] **Step 1: Fix Vite env typing**

Create `frontend/src/vite-env.d.ts` with Vite client reference and explicit env keys.

- [x] **Step 2: Split form state from API DTO**

Replace `TripFormData & { start_date: Dayjs | null; end_date: Dayjs | null }` with a local `TripFormState` type that omits date strings and stores Dayjs values.

- [x] **Step 3: Align frontend types with backend models**

Make optional nullable fields match Pydantic: attraction/hotel `location` can be absent/null, hotel `rating` is number/null, add `photos`, `poi_id`, and `price_text`.

- [x] **Step 4: Use configured API base URL for POI images**

Expose a small API helper for POI photo URLs or reuse exported base URL instead of hardcoded `http://localhost:8000`.

- [x] **Step 5: Verify**

Run: `npm run build`
Expected: PASS.

### Task 5: Repository Hygiene for Reproducibility

**Files:**
- Modify: `.gitignore`

- [x] **Step 1: Fix overbroad ignore rules**

Stop ignoring all Markdown, JSON, JSONL, SVG, PNG, and `package-lock.json` globally. Keep secrets, build outputs, caches, and dependency directories ignored.

- [x] **Step 2: Verify tracked/untracked intent**

Run: `git status --short --ignored`
Expected: source docs, eval cases, and lockfiles are visible to Git; `.env`, `node_modules`, caches remain ignored.

### Task 6: Final Verification

**Files:**
- No new files unless fixes require them.

- [x] **Step 1: Run backend tests**

Run: `pytest tests/ -v`
Expected: PASS or dependency installation blocker explicitly reported.

- [x] **Step 2: Run backend lint**

Run: `ruff check .`
Expected: PASS or real lint findings explicitly reported.

- [x] **Step 3: Run frontend build**

Run: `npm run build`
Expected: PASS.

# Workflow Truthfulness And Failure Semantics Design

## Scope

This specification covers the first remediation batch from the adversarial review:

- stop planner retry exhaustion from looping back into the planner;
- remove partial-data planning when weather or hotel acquisition fails;
- require complete weather coverage for the requested dates;
- constrain planner output to facts returned by the acquisition agents;
- validate the final itinerary before it leaves the workflow.

API authentication, global timeouts, evaluator redesign, frontend security, export behavior, and deployment controls are separate remediation batches.

## Decision

The workflow is fail-closed. It returns a complete, source-constrained `TripPlan` or raises a clear error. It does not return a degraded plan and does not ask the planner Agent to invent fields that are missing from MCP-backed acquisition results.

The current `TripPlan` response contract remains unchanged. No nullable weather, hotel, or degraded-result fields are introduced in this batch.

## Responsibility Boundary

Acquisition nodes own facts:

- attraction identity, name, address, coordinates, category, description, ticket data, and POI ID;
- hotel name, address, coordinates, rating, type, price range, and estimated cost;
- weather date and forecast fields.

The planner Agent owns arrangement only:

- which acquired attractions appear on each day;
- which acquired hotel is selected;
- meal suggestions, daily description, and transportation narrative;
- overall suggestions and budget estimates.

The planner may reference source entities but may not create or mutate them. After parsing, the workflow resolves planner references against the acquired source lists and replaces planner-supplied attraction, hotel, and weather facts with deep copies of the authoritative source objects.

## Source Resolution

Attractions are resolved by POI ID when both sides provide one. Otherwise they are resolved by a normalized name consisting of trimmed, case-folded text with internal whitespace collapsed. Hotels are resolved by the same normalized-name rule because the current `Hotel` model has no source ID.

Resolution must be unique. An unknown reference, ambiguous normalized name, duplicate attraction within one day, or missing daily hotel invalidates the planner result and triggers the normal planner retry path.

This design does not claim cryptographic provenance from the raw MCP protocol. It prevents the planner stage from introducing facts outside the acquisition-stage result set. Capturing tool-call evidence and raw provider records is a later observability enhancement.

## Request And Date Invariants

The requested date sequence is the inclusive range from `start_date` through `end_date`. Before invoking the planner, the workflow requires:

- valid ISO calendar dates;
- `start_date <= end_date`;
- inclusive date count equal to `travel_days`;
- exactly one weather record for every requested date;
- no duplicate or out-of-range weather dates;
- at least one acquired attraction and at least one acquired hotel.

If the weather provider cannot cover the requested range, the workflow fails explicitly after the weather node exhausts its retry budget. It must not substitute current weather, omit dates, or fabricate a forecast.

## Final Plan Invariants

A parsed plan is accepted only when all of the following hold:

- `city`, `start_date`, and `end_date` equal the request;
- the number of days equals `travel_days`;
- day dates exactly match the requested date sequence in order;
- `day_index` is the zero-based contiguous sequence `0..travel_days-1`;
- every day has at least one source-resolved attraction and no duplicate attraction;
- every day has one source-resolved hotel;
- every day contains exactly one `breakfast`, `lunch`, and `dinner` meal; extra `snack` entries are allowed;
- final `weather_info` is the validated acquisition result in requested-date order.

Meal type comparison is trimmed and case-insensitive. Invalid planner output raises a validation error inside the planner node, allowing at most two planner retries.

## Error State Machine

Each failed node is attempted once plus at most two retries.

- Attraction, weather, or hotel exhaustion terminates the workflow with the last concrete error.
- Planner parsing, source-resolution, or invariant-validation exhaustion terminates the workflow with the last concrete error.
- The error handler never clears an exhausted failure and never routes an exhausted node to `plan_itinerary`.
- `skip_to_plan` is removed from the graph and routing table.
- A successful retry clears the transient error through the existing state reducers and continues along the normal graph edges.

This produces a finite upper bound on node execution and prevents LangGraph's recursion limit from acting as business-level error handling.

## Testing Strategy

Tests follow red-green-refactor and use mocked Agents only at the external Agent boundary.

1. Recovery tests reproduce planner retry exhaustion and assert exactly three planner calls followed by the original planning error, without `GraphRecursionError`.
2. Recovery tests assert exhausted weather and hotel failures terminate and never invoke the planner, even when attractions are available.
3. Context tests reject missing, duplicate, out-of-range, and incomplete weather dates.
4. Planner validation tests reject wrong city, dates, day count, day order, `day_index`, meals, unknown attractions, unknown hotels, duplicate daily attractions, and missing hotels.
5. Planner validation tests prove altered model-supplied addresses and coordinates are replaced with acquired source values.
6. A valid end-to-end mocked workflow covers all requested dates and proves the accepted plan contains only source objects.
7. The full backend test suite and Ruff run after the focused tests pass.

## Compatibility And Non-Goals

- The HTTP response schema and frontend TypeScript types do not change.
- No fake fallback records are added.
- No fuzzy geographic matching or external place database is added.
- No authentication, rate limiting, timeout, cancellation, or evaluator changes are included in this batch.
- No changes are made to `.env` files or provider credentials.

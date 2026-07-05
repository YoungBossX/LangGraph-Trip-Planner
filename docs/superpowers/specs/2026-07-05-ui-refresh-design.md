# UI Refresh Design: De-AI Travel Planner

Date: 2026-07-05

## Goal

Refresh the frontend so it feels like a real travel planning tool rather than an AI demo page. The product should stay efficient, calm, and trustworthy while still acknowledging the multi-agent workflow as a quiet technical backing.

## Approved Direction

Use the "Travel Consultant Workspace" direction with a structured UI refresh across both pages:

- Home page becomes a focused trip-request workspace.
- Result page becomes an itinerary review workspace.
- Keep the existing Vue 3, Ant Design Vue, routing, API calls, SSE progress flow, map loading, edit mode, sessionStorage, and export behavior.
- Do not introduce a large component-system rewrite in this phase.

## Copy And Product Voice

The interface should not lead with "AI" or "multi-Agent" in the primary title.

Approved voice:

- Main title focuses on planning value, such as "让行程先变得清楚".
- Multi-agent is mentioned once as a technical trust cue, such as "多 Agent 工作流支持景点、天气、住宿与行程编排".
- Remove hype phrases like "智能", "完美无忧", "开始规划我的旅行", and emoji-heavy progress messages.
- Prefer direct action labels: "生成行程", "保存修改", "导出 PDF", "返回首页".

## Visual System

Approved palette: Graphite / Teal / Coral.

- Page background: `#f6f8f7`
- Surface: `#ffffff`
- Primary text / graphite: `#1f2a28`
- Secondary text: `#66736f`
- Border: `#dfe5e2`
- Soft fill: `#eef2f0`
- Primary action / teal: `#0f766e`
- Teal soft state: `#e8f5f2`
- Alert/accent / coral: `#f0765f`
- Coral soft state: `#fff7f4`

Rules:

- No large purple/blue gradients.
- No floating circles, bokeh blobs, bouncing emoji, or decorative animation.
- Shadows should be subtle and functional.
- Cards use 8px radius or less unless Ant Design internals require otherwise.
- Coral is limited to small warnings, prices, weather risks, and accents. It must not become a dominant brand color.
- Avoid card nesting except for repeated items or genuine tool panels.

## Home Page Design

Structure:

- Use a full-page workspace layout with a compact top navigation/brand strip.
- Main area is a two-column layout on desktop:
  - Left: concise product explanation and trust cues.
  - Right: trip request form.
- On mobile, stack content with the form first after the page heading.

Form changes:

- Keep current data model and validation.
- Replace emoji prefixes in inputs/selects with plain labels or Ant Design/icon-based affordances.
- Group fields into practical sections:
  - Destination and dates
  - Travel preferences
  - Additional requirements
- Convert preference tags into quiet selectable pills using the approved palette.
- Travel days should be a compact derived field, not a bright gradient badge.

Loading state:

- Keep SSE progress mapping.
- Progress text should be plain and operational:
  - "正在检索景点"
  - "正在核对天气"
  - "正在匹配住宿"
  - "正在编排行程"
  - "正在恢复流程"
- Progress bar uses teal, without gradient stroke.

## Result Page Design

Structure:

- Keep side navigation on desktop but make it quieter and narrower.
- Top header presents:
  - City and date range
  - Brief summary of days, attractions, hotels, and weather availability when available
  - Action buttons for edit/save/cancel/export
- Main content uses a review layout:
  - Summary and budget metrics near the top
  - Daily itinerary as the primary reading column
  - Map and weather as supporting panels

Daily itinerary:

- Replace emoji-heavy section labels with plain section headings.
- Attractions should read like a timeline/list rather than decorative cards competing with each other.
- Keep photos, but remove gradient fallback thumbnails. Fallbacks should be neutral placeholders.
- In edit mode, keep the existing local editing behavior but make controls compact and readable.

Map:

- Keep AMap integration and existing marker/route behavior.
- Marker labels should use teal, with coral only for selected or warning-like states.
- The map card should be framed as a functional panel, not a decorative hero block.

Weather and budget:

- Weather cards use subdued surfaces and small status accents.
- Budget metrics use compact tiles with clear labels and numbers.
- Avoid bright gradient card headers.

Export:

- Update `applyExportStyles()` so exported image/PDF matches the refreshed visual system.
- Export must remain compatible with the existing html2canvas/jsPDF flow.

## Responsive Requirements

- Desktop: two-column home layout and side-nav result layout.
- Tablet/mobile:
  - Home stacks cleanly.
  - Result page hides or reflows side navigation into top anchors/tabs if needed.
  - Cards and buttons must not overflow.
  - Text should wrap naturally without viewport-scaled font sizes.

## Technical Scope

Files expected to change during implementation:

- `frontend/src/App.vue`
- `frontend/src/views/Home.vue`
- `frontend/src/views/Result.vue`

Optional only if implementation needs shared tokens:

- Add a small frontend style token section inside existing scoped styles, or create a lightweight shared CSS file if it clearly reduces duplication.

Out of scope:

- Backend changes.
- API/schema changes.
- New routing.
- New data fields.
- Replacing Ant Design Vue.
- Full component extraction or design-system rewrite.

## Verification

Implementation should be verified with:

- `npm run build` from `frontend`
- Manual browser check for `/` and `/result`
- If sample trip data is needed for `/result`, use sessionStorage-compatible local data and do not commit fake runtime data.
- Visual checks at desktop and mobile widths for:
  - No overlapping text
  - No horizontal overflow
  - Buttons readable
  - Map panel visible when API key is configured
  - Export controls still reachable

## Acceptance Criteria

- The UI no longer reads as an AI-generated template.
- Purple gradients, decorative floating circles, bouncing icons, and emoji-heavy labels are removed.
- Multi-agent is present only as a subtle technical trust cue.
- Home page is clearly an efficient trip request tool.
- Result page is clearly an itinerary review workspace.
- Existing trip generation, streaming progress, result rendering, edit mode, map behavior, and export behavior remain intact.

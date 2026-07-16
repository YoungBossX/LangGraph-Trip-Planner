# Home Inspiration Design

**Status:** Design approved; implemented in placeholder mode.

> **2026-07-15 override:** The user approved placeholder mode because the image CDN was unavailable. The four local image paths remain reserved with `imageAvailable: false`; the UI must render a CSS/text placeholder without requesting absent files, and real image sources plus attribution must be added when assets are supplied. This supersedes the prior requirement to acquire images during this implementation.

> **2026-07-15 asset integration:** Source-cleared local WebP assets are now present as four responsive pairs: each city has a 1600 x 1200 `city.webp` source and an 800 x 600 `city-800.webp` source. Presets retain the 1600px `imageSrc`, expose both sources through `imageSrcSet`, and keep `imageAvailable` enabled. Attribution is recorded in `frontend/public/inspiration/ATTRIBUTIONS.md`; the fallback behavior remains required for future load failures.

## Goal

Turn the homepage from a sparse, single-purpose form into a domestic-travel planning workspace with enough useful content to sustain the page. The page should create a sense of travel through destination-focused presentation and actionable trip examples, while keeping plan creation as the primary task. Real city imagery will replace the initial placeholders after source-cleared assets and attribution are supplied.

The visual tone remains white, graphite, and cold blue. The page must not become a generic AI product landing page or an information-only travel magazine.

## Scope

### In scope

- Replace the current text-and-metrics hero column with a destination feature surface and concise trip context; use the approved city-label placeholder until real imagery is supplied.
- Add four domestic destination presets: Hangzhou, Beijing, Shanghai, and Changsha.
- Make a preset prefill compatible form fields without changing the backend API.
- Add a restrained, full-width planning-process band below the presets.
- Reserve four local image paths and render responsive CSS/text placeholders without requesting unavailable files.
- Preserve the existing streaming request, form validation, date restrictions, and result-page workflow.
- Carry forward the previously approved global and result-page neutral palette alignment in the same UI delivery, without changing result data or layout behavior.

### Out of scope

- New backend routes, image-search APIs, user accounts, saved presets, or database storage.
- Changes to generated itinerary data, AMap requests, SSE behavior, or result-page layout and data behavior beyond the approved palette alignment.
- Invented attractions, hotels, or weather data. Presets describe planning preferences only; generated trip content remains sourced by the existing workflow.

## Page Structure

### Desktop

1. Keep the compact top navigation on a constrained page width.
2. Use a stable two-column hero below it:
   - Left: a 16:10 Hangzhou destination surface with city, short route mood, and recommended duration; it uses a city-label placeholder now and an editorial image after an attributed asset is supplied.
   - Right: the existing trip request form, still presented as the primary interactive surface.
3. Follow the hero with a full-width `从一个目的地开始` section containing four individual preset tiles.
4. Follow the preset section with a quiet, light-neutral process band: choose a preset, confirm dates, retrieve real place/weather data, edit the itinerary.

The desktop page should naturally extend to roughly 1.5--2 viewports rather than leaving the lower half of a wide monitor empty.

### Mobile

1. Stack the destination media surface before the form.
2. Preserve readable media and form proportions; neither may be compressed by text or controls.
3. Render the four preset tiles in a horizontal, scrollable rail with fixed tile dimensions.
4. Render the process band as a compact vertical sequence.

## Presets

| City | Preset title | Recommended duration | Interests |
| --- | --- | ---: | --- |
| Hangzhou | 湖畔慢游 | 3 days | History and culture, natural scenery, leisure |
| Beijing | 古都与新展 | 4 days | History and culture, art |
| Shanghai | 城市漫游 | 3 days | Art, shopping, food |
| Changsha | 晚风与夜宵 | 3 days | Food, leisure |

Each preset also defines a supported transportation option, accommodation preference, image path, alternative text, and a concise description. Preset copy is illustrative guidance only; it must not imply that specific itinerary entries were pre-generated.

## Prefill Behavior

- A preset tile is a semantic interactive control and shows a selected state after it is chosen.
- Selecting a preset updates `city`, `transportation`, `accommodation`, and `preferences` in the existing form.
- The form does not submit automatically, and selecting a preset never clears user-entered dates.
- `recommendedDays` is not sent as an authoritative itinerary duration. Until dates exist, the duration UI shows it as a recommendation.
- When a selected preset exists and a start date exists but an end date does not, calculate the suggested end date as `start + recommendedDays - 1`, subject to the existing 30-day limit. This applies whether the preset or the start date was selected first.
- If both dates already exist, they remain unchanged. The existing date watcher continues to compute the actual `travel_days` used by the API.
- Any request still requires the user's confirmed start and end dates and follows the existing validation and SSE flow.

## Visual System

- Primary surfaces: `#ffffff`, `#fafafa`, and graphite text `#111111`.
- Borders: neutral gray rather than colored panels.
- Interaction and selected states: cold blue `#165dff`; hover uses the existing darker blue.
- Source-cleared real photography will supply warmth after assets are provided. Until then, use restrained neutral placeholders without large gradients, AI-themed decorative graphics, floating page-section cards, or prominent technical claims.
- Individual preset tiles may use an 8px-or-smaller radius. The hero media surface should read as a destination feature, not a decorative card.
- The process band is a full-width layout band, not a collection of nested cards.

## Image and Loading Strategy

- The eight `/inspiration/*.webp` paths form four source-cleared city pairs with attribution in `frontend/public/inspiration/ATTRIBUTIONS.md`: each base path is a 1600 x 1200 source and each `-800` path is an 800 x 600 source. Presets expose `imageAvailable: true` and an `imageSrcSet` containing both variants.
- Render an `<img>` only when its corresponding `imageAvailable` flag is `true`. If an asset becomes unavailable in a future environment, retain the neutral CSS/text city-label fallback so the page remains usable without a broken image.
- Future asset replacements must update the local file and its attribution record before changing the corresponding flag or metadata.
- Once available, the Hangzhou hero image loads eagerly and lower preset images use lazy loading. Both image locations bind the preset `imageSrcSet`; feature `sizes` describes its desktop column and tile `sizes` selects the 800px source at card widths. Target approximately 300 KB per WebP and use fixed aspect ratios with `object-fit: cover` to prevent layout shift.
- If an available image fails to load, show the same city-label fallback while preserving contrast and preset click behavior.
- Do not fetch images at runtime or add an external image API dependency.

## Accessibility and Resilience

- Every available image has city-specific alternative text; every unavailable slot exposes its city label as text.
- Preset tiles are keyboard reachable and expose a clear selected state.
- Text overlays maintain contrast against photography through a restrained solid scrim where necessary.
- At desktop and mobile widths, copy wraps cleanly and controls retain stable dimensions.
- An unavailable or failed asset must not block form input, preset selection, validation, or form submission.

## Implementation Boundaries

The initial implementation should remain closely scoped:

- `frontend/src/views/Home.vue`: layout, preset configuration, prefill handler, responsive styles, and loading-state integration.
- `frontend/src/data/tripPresets.ts`: the four reserved local paths and their `imageAvailable: false` integration flags.
- `frontend/src/App.vue`: previously approved global neutral theme tokens.
- `frontend/src/views/Result.vue`: previously approved neutral/blue palette alignment only; result layout and data flow remain unchanged.
- `frontend/public/inspiration/`: deferred until the user supplies real files and source attribution; no assets are created during placeholder-mode implementation.
- This specification document.

No separate component is necessary for four static presets; adding one would increase indirection without reducing meaningful complexity.

## Verification

1. Select each preset and confirm that city, transportation, accommodation, and interests are populated correctly.
2. Confirm all date cases: no dates, start date only, both dates set, and a 30-day boundary.
3. Confirm every `imageAvailable: false` slot renders a city-label placeholder without a WebP network request; once an asset is integrated, confirm a load failure returns to the same usable fallback.
4. Inspect the page at desktop and mobile widths for no clipping, unexpected horizontal scroll, overlap, or whitespace imbalance.
5. Run `npm run build` in `frontend` after implementation.

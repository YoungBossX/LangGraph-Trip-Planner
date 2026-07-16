# Wide Home Workbench Design

**Status:** Approved for implementation on 2026-07-15.

## Goal

Make the desktop homepage feel like a focused travel-planning workspace rather than a narrow form floating in empty space. The primary action remains creating a trip plan; the destination imagery and selected-preset state provide travel context without introducing marketing metrics or decorative cards.

## Layout

- Use one 1440px desktop content rail for navigation, the hero/form workspace, destination presets, and the workflow band.
- Keep the two-column workspace, with the destination feature at roughly 42% and the planning form at roughly 58%. Use a 32px gap and align both columns at their top edges.
- Let the form determine its own height. The destination feature uses a 560px minimum height on desktop instead of stretching to match the form.
- Keep the existing one-column workspace and horizontal preset rail at tablet and mobile widths.

## Form Hierarchy

1. Keep the existing city, start date, end date, and actual/recommended days controls together.
2. Place transportation and accommodation in two compact columns, with interests using the remaining half of the row so selectable preferences scan as one group.
3. Reduce the optional request textarea to two visible rows.
4. When a preset is selected, show an inline plan summary above the submission button. It contains the selected city, actual or recommended duration, transportation, and accommodation. It is a bordered information row, not another card.

## Image Treatment

- Retain the source-cleared local responsive image pairs, their `srcset` delivery, and their guarded fallback behavior.
- Keep the featured Hangzhou image darkened by the existing solid scrim. Position the image slightly left of center so willow detail remains visible behind the text.
- Do not add gradients, technical metrics, another sidebar, or a third workspace column.

## Verification

- A pure helper returns the four summary values, choosing actual days when present and the preset recommendation otherwise.
- Existing preset/date tests, type checking, and the production build pass.
- The production build copies all eight responsive WebP variants and `ATTRIBUTIONS.md` into `dist/inspiration`.

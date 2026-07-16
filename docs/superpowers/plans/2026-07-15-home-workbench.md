# Wide Home Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the homepage into a balanced desktop travel-planning workbench while exposing a useful selected-preset summary above the existing generation action.

**Architecture:** Keep page layout and form state in `Home.vue`. Put the display-only plan summary decision in a pure utility so actual versus recommended duration is testable without mounting the page. Preserve the existing SSE request, validation, and responsive mobile layout.

**Tech Stack:** Vue 3 Composition API, TypeScript, Ant Design Vue, Vitest, and scoped CSS.

---

## File Structure

- Create: `frontend/src/utils/planSummary.ts`
  - Returns the four display values for a selected preset summary.
- Create: `frontend/src/utils/planSummary.test.ts`
  - Covers actual-duration and recommended-duration summary states.
- Modify: `frontend/src/views/Home.vue`
  - Uses the helper, rebalances the form grid, renders the inline summary, and widens the desktop workspace.

### Task 1: Add a Testable Plan Summary Helper

**Files:**
- Create: `frontend/src/utils/planSummary.test.ts`
- Create: `frontend/src/utils/planSummary.ts`

- [ ] **Step 1: Write the failing summary tests**

```ts
import { describe, expect, it } from 'vitest'
import { getPlanSummaryItems } from './planSummary'

describe('getPlanSummaryItems', () => {
  it('uses the selected preset duration until real dates provide a duration', () => {
    expect(getPlanSummaryItems({
      city: '杭州',
      travelDays: 0,
      recommendedDays: 3,
      transportation: '公共交通',
      accommodation: '舒适型酒店',
    })).toEqual(['杭州', '3 天建议', '公共交通', '舒适型酒店'])
  })

  it('uses an actual duration once dates have produced one', () => {
    expect(getPlanSummaryItems({
      city: '北京',
      travelDays: 4,
      recommendedDays: 3,
      transportation: '公共交通',
      accommodation: '舒适型酒店',
    })).toEqual(['北京', '4 天行程', '公共交通', '舒适型酒店'])
  })
})
```

- [ ] **Step 2: Run the focused test and confirm it fails because the helper does not exist**

Run: `npm run test -- src/utils/planSummary.test.ts`

Expected: the test runner reports that `./planSummary` cannot be resolved.

- [ ] **Step 3: Implement the helper**

```ts
export interface PlanSummaryInput {
  city: string
  travelDays: number
  recommendedDays: number
  transportation: string
  accommodation: string
}

export const getPlanSummaryItems = ({
  city,
  travelDays,
  recommendedDays,
  transportation,
  accommodation,
}: PlanSummaryInput): readonly string[] => [
  city,
  travelDays > 0 ? `${travelDays} 天行程` : `${recommendedDays} 天建议`,
  transportation,
  accommodation,
]
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run: `npm run test -- src/utils/planSummary.test.ts`

Expected: 2 passing tests.

### Task 2: Rebalance the Homepage Workspace and Form

**Files:**
- Modify: `frontend/src/views/Home.vue`

- [ ] **Step 1: Import the summary helper and expose computed values**

```ts
import { getPlanSummaryItems } from '@/utils/planSummary'

const planSummaryItems = computed(() => {
  if (!selectedPreset.value) {
    return []
  }

  return getPlanSummaryItems({
    city: formData.city,
    travelDays: formData.travel_days,
    recommendedDays: selectedPreset.value.recommendedDays,
    transportation: formData.transportation,
    accommodation: formData.accommodation,
  })
})
```

- [ ] **Step 2: Rebalance the preference controls and add the inline summary**

```vue
<a-col :xs="24" :md="6">...</a-col>
<a-col :xs="24" :md="6">...</a-col>
<a-col :xs="24" :md="12">...</a-col>

<div v-if="planSummaryItems.length" class="plan-summary" role="status" aria-live="polite">
  <span class="plan-summary__label">当前计划</span>
  <span v-for="item in planSummaryItems" :key="item" class="plan-summary__item">{{ item }}</span>
</div>
```

Set the optional textarea to `:rows="2"`. Insert the summary after the optional request section and before the submit form item. Do not modify form submission, date watchers, or SSE state.

- [ ] **Step 3: Apply the desktop and mobile scoped CSS rules**

```css
.home-nav,
.home-workspace,
.inspiration-section,
.workflow-band {
  max-width: 1440px;
}

.home-workspace {
  grid-template-columns: minmax(480px, 0.82fr) minmax(640px, 1.18fr);
  gap: 32px;
  align-items: start;
}

.destination-feature,
.destination-feature__content {
  min-height: 560px;
}

.destination-feature__image {
  object-position: 36% center;
}

.plan-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
  padding: 14px 0;
  border-top: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
}

.plan-summary__label {
  color: #111111;
  font-size: 13px;
  font-weight: 800;
}

.plan-summary__item {
  padding-left: 10px;
  border-left: 1px solid #d9d9d9;
  color: #666666;
  font-size: 13px;
}
```

Keep the current `max-width: 900px` one-column rule. At `max-width: 640px`, allow summary items to wrap cleanly without the border-left on the first label.

- [ ] **Step 4: Run homepage verification**

Run: `npm run test`, `npx vue-tsc --noEmit`, and `npm run build` from `frontend`.

Expected: all tests pass, type checking exits 0, and the production build copies all eight responsive WebP variants and `ATTRIBUTIONS.md` into `dist/inspiration`.

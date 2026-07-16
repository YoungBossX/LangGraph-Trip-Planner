# Home Inspiration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a useful, image-led domestic destination inspiration experience to the homepage while preserving the existing trip-planning form and SSE workflow.

**Architecture:** Keep layout and form integration in `Home.vue`, because the page owns the form state already. Put static preset content and the date suggestion helper in focused TypeScript modules so the date behavior is testable without a browser. Reserve four Vite `public` paths behind `imageAvailable` flags; render local CSS/text placeholders now, then enable local WebP files only after the user supplies real assets and attribution.

**Tech Stack:** Vue 3 Composition API, TypeScript, Ant Design Vue, Day.js, Vite, Vitest, CSS/text image placeholders, and deferred local WebP assets.

> **2026-07-15 override:** The user approved placeholder mode because the image CDN was unavailable. The four local image paths remain reserved with `imageAvailable: false`; the UI must render a CSS/text placeholder without requesting absent files, and real image sources plus attribution must be added when assets are supplied. This supersedes the prior requirement to acquire images during this implementation.

> **2026-07-15 asset integration:** The user supplied source-cleared local WebP assets as four responsive pairs: 1600 x 1200 `city.webp` sources and 800 x 600 `city-800.webp` sources. Their attribution is recorded in `frontend/public/inspiration/ATTRIBUTIONS.md`; all corresponding `imageAvailable` flags are now `true`, and `imageSrcSet` exposes both variants.

---

## File Structure

- Create: `frontend/src/data/tripPresets.ts`
  - Defines the four domestic destination presets and their form-compatible values.
- Create: `frontend/src/utils/presetDates.ts`
  - Exposes a pure, bounded end-date suggestion helper.
- Create: `frontend/src/utils/presetDates.test.ts`
  - Covers the preset catalog and all date-suggestion branches.
- Deferred until the user supplies real files and attribution; do not create during placeholder-mode implementation:
  - `frontend/public/inspiration/hangzhou.webp`
  - `frontend/public/inspiration/beijing.webp`
  - `frontend/public/inspiration/shanghai.webp`
  - `frontend/public/inspiration/changsha.webp`
  - `frontend/public/inspiration/ATTRIBUTIONS.md`
- Modify: `frontend/package.json`
  - Adds a deterministic frontend test command and Vitest development dependency.
- Modify: `frontend/package-lock.json`
  - Locks the Vitest dependency graph after installation.
- Modify: `frontend/src/views/Home.vue:1-483`
  - Replaces the sparse hero panel, connects preset selection to the existing form, adds the inspiration and workflow sections, and implements responsive styles.
- Carried forward from the previously approved palette pass: `frontend/src/App.vue` and `frontend/src/views/Result.vue`
  - Aligns global and result-page colors with the white, graphite, neutral, and cold-blue visual system without changing result behavior.

## Task 1: Establish Testable Preset Data and Date Suggestions

**Files:**
- Create: `frontend/src/data/tripPresets.ts`
- Create: `frontend/src/utils/presetDates.ts`
- Create: `frontend/src/utils/presetDates.test.ts`
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`

- [ ] **Step 1: Add Vitest and the test command**

Run:

```powershell
cd frontend
npm install --save-dev vitest
```

Add this script to `frontend/package.json`:

```json
"test": "vitest run"
```

Expected: `package.json` lists `vitest` under `devDependencies`, and `package-lock.json` records the resolved package graph.

- [ ] **Step 2: Write the failing preset/date tests**

Create `frontend/src/utils/presetDates.test.ts`:

```ts
import dayjs from 'dayjs'
import { describe, expect, it } from 'vitest'
import { tripPresets } from '@/data/tripPresets'
import { suggestPresetEndDate } from '@/utils/presetDates'

describe('trip presets', () => {
  it('exposes the approved domestic cities in display order', () => {
    expect(tripPresets.map((preset) => preset.city)).toEqual(['杭州', '北京', '上海', '长沙'])
  })

  it('keeps every recommended duration within the supported form limit', () => {
    expect(tripPresets.every((preset) => preset.recommendedDays >= 1 && preset.recommendedDays <= 30)).toBe(true)
  })
})

describe('suggestPresetEndDate', () => {
  it('suggests an inclusive end date when only the start date exists', () => {
    expect(suggestPresetEndDate(dayjs('2026-08-01'), null, 3)?.format('YYYY-MM-DD')).toBe('2026-08-03')
  })

  it('does not replace an existing end date', () => {
    expect(suggestPresetEndDate(dayjs('2026-08-01'), dayjs('2026-08-05'), 3)).toBeNull()
  })

  it('does not suggest a date without a start date or for an invalid duration', () => {
    expect(suggestPresetEndDate(null, null, 3)).toBeNull()
    expect(suggestPresetEndDate(dayjs('2026-08-01'), null, 0)).toBeNull()
    expect(suggestPresetEndDate(dayjs('2026-08-01'), null, 31)).toBeNull()
  })
})
```

- [ ] **Step 3: Run the test to verify it fails before implementation**

Run:

```powershell
cd frontend
npm run test -- src/utils/presetDates.test.ts
```

Expected: FAIL because `@/data/tripPresets` and `@/utils/presetDates` do not exist.

- [ ] **Step 4: Implement the preset catalog and date helper**

Create `frontend/src/data/tripPresets.ts`:

```ts
import type { TripFormData } from '@/types'

export type TripPresetId = 'hangzhou' | 'beijing' | 'shanghai' | 'changsha'

export interface TripPreset {
  readonly id: TripPresetId
  readonly city: string
  readonly title: string
  readonly description: string
  readonly recommendedDays: number
  readonly transportation: TripFormData['transportation']
  readonly accommodation: TripFormData['accommodation']
  readonly preferences: Readonly<TripFormData['preferences']>
  readonly imageSrc: string
  readonly imageAvailable: boolean
  readonly imageAlt: string
}

export const tripPresets: readonly TripPreset[] = [
  {
    id: 'hangzhou',
    city: '杭州',
    title: '湖畔慢游',
    description: '把湖光、旧街与松弛的步调留给一个周末。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['历史文化', '自然风光', '休闲'],
    imageSrc: '/inspiration/hangzhou.webp',
    imageAvailable: false,
    imageAlt: '杭州湖畔与城市天际线',
  },
  {
    id: 'beijing',
    city: '北京',
    title: '古都与新展',
    description: '在城墙、展馆与街巷之间读一座城市。',
    recommendedDays: 4,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['历史文化', '艺术'],
    imageSrc: '/inspiration/beijing.webp',
    imageAvailable: false,
    imageAlt: '北京城市建筑与传统景观',
  },
  {
    id: 'shanghai',
    city: '上海',
    title: '城市漫游',
    description: '把展览、街区与一顿好饭排进同一天。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['艺术', '购物', '美食'],
    imageSrc: '/inspiration/shanghai.webp',
    imageAvailable: false,
    imageAlt: '上海城市天际线与滨水街景',
  },
  {
    id: 'changsha',
    city: '长沙',
    title: '晚风与夜宵',
    description: '把白天的松弛和夜里的烟火一起安排。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '经济型酒店',
    preferences: ['美食', '休闲'],
    imageSrc: '/inspiration/changsha.webp',
    imageAvailable: false,
    imageAlt: '长沙夜景与街头烟火',
  },
]
```

Create `frontend/src/utils/presetDates.ts`:

```ts
import type { Dayjs } from 'dayjs'

export const suggestPresetEndDate = (
  startDate: Dayjs | null,
  endDate: Dayjs | null,
  recommendedDays: number,
): Dayjs | null => {
  if (!startDate || endDate || !Number.isInteger(recommendedDays) || recommendedDays < 1 || recommendedDays > 30) {
    return null
  }

  return startDate.add(recommendedDays - 1, 'day')
}
```

- [ ] **Step 5: Run the focused tests and type check**

Run:

```powershell
cd frontend
npm run test -- src/utils/presetDates.test.ts
npx vue-tsc --noEmit
```

Expected: all five Vitest tests pass and `vue-tsc` exits with code 0.

- [ ] **Step 6: Commit the isolated preset foundation**

```powershell
git add frontend/package.json frontend/package-lock.json frontend/src/data/tripPresets.ts frontend/src/utils/presetDates.ts frontend/src/utils/presetDates.test.ts
git commit -m "test: cover trip preset date suggestions"
```

## Task 2: Reserve Destination Image Slots in Placeholder Mode

**Files:**
- Modify: `frontend/src/data/tripPresets.ts`
- Modify: `frontend/src/utils/presetDates.test.ts`
- Modify: this implementation plan and its design specification

- [ ] **Step 1: Lock the reserved image contract with a focused test**

Assert that the four presets retain their exact `/inspiration/*.webp` paths and each exposes `imageAvailable: false`. The test must fail if a path changes or a flag becomes available before its file and attribution exist.

This test locks the current placeholder metadata state only. It does not inspect the filesystem or attribution record, so passing it does not by itself verify that an image file or its attribution exists.

- [ ] **Step 2: Add the availability flag without creating assets**

Add readonly `imageAvailable: boolean` to `TripPreset` and set it to `false` for Hangzhou, Beijing, Shanghai, and Changsha. Keep the catalog values and reserved paths unchanged. Do not make a network request, download a file, or create a fake image asset.

- [ ] **Step 3: Enforce the placeholder integration contract**

The hero and preset tiles must create an `<img>` only when the corresponding flag is `true`. While it is `false`, render a CSS/text placeholder containing the city name so the browser never requests the absent path.

### Deferred Asset Handoff

Asset acquisition is not part of the current implementation. After the user supplies real, source-cleared files and attribution:

1. Verify each photograph depicts its named city and record the original URL, creator, license, and retrieval date in `frontend/public/inspiration/ATTRIBUTIONS.md`.
2. Crop and optimize each supplied file for the 16:10 hero and 4:3 tiles, targeting at least 1600px width and approximately 300 KB or less where quality permits.
3. Place each file at its already-reserved local path.
4. After both the real file and its attribution record are present, switch only that preset's `imageAvailable` flag to `true`.
5. In the same future change, update that preset's hard-coded `imageAvailable` expectation in `frontend/src/utils/presetDates.test.ts` from `false` to `true`.

## Task 3: Rebuild the Homepage Around the Inspiration-to-Plan Flow

**Files:**
- Modify: `frontend/src/views/Home.vue:1-483`

- [ ] **Step 1: Add the imports and preset state before changing the template**

At `frontend/src/views/Home.vue:191`, update imports and add the following state after `const router = useRouter()`:

```ts
import { computed, reactive, ref, watch } from 'vue'
import { tripPresets, type TripPreset } from '@/data/tripPresets'
import { suggestPresetEndDate } from '@/utils/presetDates'

const selectedPresetId = ref<TripPreset['id'] | null>(null)
const failedPresetImages = reactive<Partial<Record<TripPreset['id'], true>>>({})
const featuredPreset = tripPresets[0]!
const selectedPreset = computed(
  () => tripPresets.find((preset) => preset.id === selectedPresetId.value) ?? null,
)

const selectPreset = (preset: TripPreset) => {
  selectedPresetId.value = preset.id
  formData.city = preset.city
  formData.transportation = preset.transportation
  formData.accommodation = preset.accommodation
  formData.preferences = [...preset.preferences]

  const suggestedEndDate = suggestPresetEndDate(
    formData.start_date,
    formData.end_date,
    preset.recommendedDays,
  )

  if (suggestedEndDate) {
    formData.end_date = suggestedEndDate
  }
}

const markPresetImageFailed = (presetId: TripPreset['id']) => {
  failedPresetImages[presetId] = true
}
```

- [ ] **Step 2: Extend the existing date watcher without changing validation semantics**

At `frontend/src/views/Home.vue:232`, add the selected-preset branch at the start of the existing watcher callback:

```ts
watch([() => formData.start_date, () => formData.end_date], ([start, end]) => {
  if (start && !end && selectedPreset.value) {
    const suggestedEndDate = suggestPresetEndDate(start, end, selectedPreset.value.recommendedDays)
    if (suggestedEndDate) {
      formData.end_date = suggestedEndDate
      return
    }
  }

  if (!start || !end) {
    formData.travel_days = 0
    return
  }

  const days = end.diff(start, 'day') + 1
  if (days > 0 && days <= 30) {
    formData.travel_days = days
  } else if (days > 30) {
    message.warning('旅行天数不能超过30天')
    formData.end_date = null
  } else {
    message.warning('结束日期不能早于开始日期')
    formData.end_date = null
  }
})
```

This preserves existing date validation, sets an actual `travel_days` only after both dates exist, and allows either selection order for a preset and start date.

- [ ] **Step 3: Replace the sparse hero template with the image-led hero and valid navigation anchors**

At `frontend/src/views/Home.vue:1-190`:

1. Replace the four non-interactive navigation spans with these anchor links:

```vue
<div class="nav-meta" aria-label="页面导航">
  <a href="#inspiration">灵感</a>
  <a href="#plan-form">规划</a>
  <a href="#workflow">流程</a>
</div>
```
2. Replace `<aside class="intro-panel">` with this featured destination block:

```vue
<aside
  class="destination-feature"
  :class="{ 'is-image-fallback': !featuredPreset.imageAvailable || failedPresetImages[featuredPreset.id] }"
>
  <img
    v-if="featuredPreset.imageAvailable && !failedPresetImages[featuredPreset.id]"
    :src="featuredPreset.imageSrc"
    :alt="featuredPreset.imageAlt"
    class="destination-feature__image"
    @error="markPresetImageFailed(featuredPreset.id)"
  />
  <div v-else class="destination-feature__placeholder">
    <span>{{ featuredPreset.city }}</span>
  </div>
  <div class="destination-feature__scrim" aria-hidden="true"></div>
  <div class="destination-feature__content">
    <p class="eyebrow">从一座城市开始</p>
    <h1>{{ featuredPreset.city }}，{{ featuredPreset.title }}</h1>
    <p>{{ featuredPreset.description }}</p>
    <span>{{ featuredPreset.recommendedDays }} 天推荐停留</span>
  </div>
</aside>
```

3. Add `id="plan-form"` to the request-card wrapper.
4. Replace the duration UI so it has a dedicated recommendation state when no actual dates exist:

```vue
<div class="days-pill">
  <template v-if="formData.travel_days">
    <strong>{{ formData.travel_days }}</strong>
    <span>天</span>
  </template>
  <template v-else-if="selectedPreset">
    <strong>{{ selectedPreset.recommendedDays }}</strong>
    <span>推荐天数</span>
  </template>
  <template v-else>
    <strong>0</strong>
    <span>天</span>
  </template>
</div>
```

- [ ] **Step 4: Add the preset and workflow sections after the hero workspace**

Insert the following sections immediately after the closing `</section>` for `home-workspace` and before `</main>`:

```vue
<section id="inspiration" class="inspiration-section" aria-labelledby="inspiration-title">
  <div class="section-intro">
    <p class="eyebrow">目的地灵感</p>
    <h2 id="inspiration-title">从一个目的地开始</h2>
    <p>选择一个出发方式，再按自己的时间调整细节。</p>
  </div>
  <div class="preset-grid">
    <button
      v-for="preset in tripPresets"
      :key="preset.id"
      type="button"
      class="preset-tile"
      :class="{
        'is-selected': selectedPresetId === preset.id,
        'is-image-fallback': !preset.imageAvailable || failedPresetImages[preset.id],
      }"
      :aria-pressed="selectedPresetId === preset.id"
      @click="selectPreset(preset)"
    >
      <img
        v-if="preset.imageAvailable && !failedPresetImages[preset.id]"
        :src="preset.imageSrc"
        :alt="preset.imageAlt"
        loading="lazy"
        @error="markPresetImageFailed(preset.id)"
      />
      <span v-else class="preset-tile__placeholder">{{ preset.city }}</span>
      <span class="preset-tile__scrim" aria-hidden="true"></span>
      <span class="preset-tile__content">
        <span class="preset-tile__meta">{{ preset.city }} · {{ preset.recommendedDays }} 天</span>
        <strong>{{ preset.title }}</strong>
        <span>{{ preset.description }}</span>
        <span class="preset-tile__command">带入计划</span>
      </span>
    </button>
  </div>
</section>

<section id="workflow" class="workflow-band" aria-labelledby="workflow-title">
  <div>
    <p class="eyebrow">规划依据</p>
    <h2 id="workflow-title">把灵感整理成可执行的行程</h2>
  </div>
  <ol class="workflow-steps">
    <li><strong>01</strong><span>选择目的地</span></li>
    <li><strong>02</strong><span>确认出发日期</span></li>
    <li><strong>03</strong><span>匹配真实地点与天气</span></li>
    <li><strong>04</strong><span>继续编辑行程</span></li>
  </ol>
</section>
```

- [ ] **Step 5: Replace the homepage styles with stable image and responsive layout rules**

At `frontend/src/views/Home.vue:310`, remove styles for `.intro-panel` and `.trust-grid`, then add rules with these non-negotiable values:

```css
.home-workspace {
  grid-template-columns: minmax(0, 0.95fr) minmax(560px, 1.05fr);
  align-items: stretch;
}

.home-page {
  background: #ffffff;
}

.nav-meta a {
  color: inherit;
  text-decoration: none;
}

.destination-feature {
  position: relative;
  min-height: 620px;
  overflow: hidden;
  background: #111111;
  color: #ffffff;
}

.destination-feature__image,
.preset-tile img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.destination-feature__placeholder,
.preset-tile__placeholder {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  background: #e9edf2;
  color: #4b5563;
  font-weight: 600;
}

.destination-feature__placeholder {
  font-size: 32px;
}

.preset-tile__placeholder {
  font-size: 20px;
}

.destination-feature__scrim,
.preset-tile__scrim {
  position: absolute;
  inset: 0;
  background: rgba(17, 17, 17, 0.42);
}

.destination-feature__content,
.preset-tile__content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
}

.destination-feature__content {
  min-height: 620px;
  justify-content: flex-end;
  align-items: flex-start;
  padding: clamp(24px, 4vw, 48px);
}

.preset-tile__content {
  min-height: 304px;
  justify-content: flex-end;
  padding: 20px;
}

.inspiration-section,
.workflow-band {
  max-width: 1220px;
  margin: 72px auto 0;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.preset-tile {
  position: relative;
  min-height: 304px;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #111111;
  color: #ffffff;
  cursor: pointer;
  text-align: left;
}

.preset-tile.is-selected {
  outline: 3px solid rgba(22, 93, 255, 0.28);
  border-color: #165dff;
}

.preset-tile:focus-visible {
  outline: 3px solid #165dff;
  outline-offset: 3px;
}

.workflow-band {
  display: grid;
  grid-template-columns: minmax(260px, 0.7fr) minmax(0, 1.3fr);
  gap: 36px;
  align-items: end;
  padding: 36px 0;
  border-top: 1px solid #e0e0e0;
  border-bottom: 1px solid #e0e0e0;
}

.workflow-steps {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.workflow-steps li {
  display: grid;
  gap: 8px;
  color: #555555;
}

.workflow-steps strong {
  color: #165dff;
  font-size: 13px;
}

@media (max-width: 900px) {
  .home-workspace,
  .workflow-band {
    grid-template-columns: 1fr;
  }

  .destination-feature {
    min-height: auto;
    aspect-ratio: 4 / 3;
  }

  .destination-feature__content {
    min-height: 100%;
  }

  .preset-grid {
    display: flex;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    padding-bottom: 8px;
  }

  .preset-tile {
    flex: 0 0 min(78vw, 300px);
    scroll-snap-align: start;
  }

  .workflow-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
```

Keep the existing form control styles, loading panel, and the `640px` navigation adjustments. The supplied focus, image, and content-positioning rules replace the obsolete intro-panel and trust-grid styles.

- [ ] **Step 6: Run unit tests, type checks, and the production build**

Run:

```powershell
cd frontend
npm run test
npm run build
```

Expected: all preset/date tests pass, Vue type checking passes, and Vite produces a production build.

- [ ] **Step 7: Commit the homepage behavior and styles**

```powershell
git add frontend/src/views/Home.vue
git commit -m "feat: add destination inspiration presets"
```

## Task 4: Perform Visual and Functional Acceptance Checks

**Files:**
- Modify only if a check reveals a specific defect: `frontend/src/views/Home.vue`

- [ ] **Step 1: Start the frontend in development mode**

Run:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Expected: Vite reports a local URL at `http://127.0.0.1:5173/`.

- [ ] **Step 2: Verify the desktop flow at 1440px width**

Check these facts in the browser:

1. The hero image and form share the first viewport without a large blank lower area.
2. The four destination tiles are visible beneath the workspace and use the approved city order.
3. Selecting each tile changes city, transportation, accommodation, interests, selected state, and recommendation text.
4. A selected preset plus a start date creates the inclusive suggested end date; an existing end date remains unchanged.
5. The submit button still reaches the existing streaming loading state.

- [ ] **Step 3: Verify the mobile flow at 390px width**

Check these facts in the browser:

1. The image, form, and workflow stack vertically with no horizontal page scroll.
2. Preset tiles scroll horizontally as a stable rail and retain readable text.
3. Focus rings, selected state, date inputs, and the submit button remain reachable.
4. No text overlaps or clips at the nav, image overlay, form labels, or workflow steps.

- [ ] **Step 4: Verify placeholder mode without an asset request**

With all four `imageAvailable` flags set to `false`, reload and confirm that the hero and tiles show their city-label placeholders, remain selectable, and produce no requests for the reserved `.webp` paths. After a real asset is integrated in the deferred handoff, separately verify that a load failure returns that slot to the same placeholder.

- [ ] **Step 5: Run final checks and commit any defect correction**

Run:

```powershell
cd frontend
npm run test
npm run build
git diff --check
```

Expected: all commands exit with code 0. If a visual or interaction defect was corrected, commit only the corrected files:

```powershell
git add frontend/src/views/Home.vue
git commit -m "fix: polish inspiration homepage behavior"
```

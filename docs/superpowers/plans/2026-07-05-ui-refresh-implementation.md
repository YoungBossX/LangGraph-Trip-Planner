# UI Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the Vue frontend so the home and result pages feel like a calm travel planning workspace instead of an AI demo.

**Architecture:** Keep the current Vue 3 + Ant Design Vue single-file view structure. Make focused edits to the app shell, `Home.vue`, and `Result.vue`, preserving the existing API calls, SSE stream handling, sessionStorage flow, map integration, edit mode, and export flow.

**Tech Stack:** Vue 3, TypeScript, Vite, Ant Design Vue, AMap JS API, html2canvas, jsPDF.

---

## File Structure

- Modify `frontend/src/App.vue`: remove the default Ant Design marketing shell and define app-level base typography/background behavior.
- Modify `frontend/src/views/Home.vue`: replace decorative hero/card styling with the Graphite / Teal / Coral trip-request workspace, keeping the existing form state and submit logic.
- Modify `frontend/src/views/Result.vue`: rebuild visual hierarchy into an itinerary review workspace, add small computed summary helpers, refresh map/photo/export styling, and keep existing editing/export behavior intact.
- Do not modify backend files or frontend API/types unless TypeScript reveals a real mismatch.

---

### Task 1: App Shell Cleanup

**Files:**
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Replace the Ant Design layout wrapper**

Replace the current template with a plain router shell so each page owns its own layout:

```vue
<template>
  <router-view />
</template>
```

- [ ] **Step 2: Replace global app styles**

Use this style block in `frontend/src/App.vue`:

```vue
<style>
:root {
  color: #1f2a28;
  background: #f6f8f7;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 320px;
  min-height: 100vh;
  background: #f6f8f7;
}

#app {
  min-height: 100vh;
}
</style>
```

- [ ] **Step 3: Verify app shell compiles**

Run from `frontend`:

```bash
npm run build
```

Expected: build passes. Existing Vite large chunk warnings are acceptable.

- [ ] **Step 4: Commit app shell cleanup**

```bash
git add frontend/src/App.vue
git commit -m "style: simplify frontend app shell"
```

---

### Task 2: Home Page Workspace Refresh

**Files:**
- Modify: `frontend/src/views/Home.vue`

- [ ] **Step 1: Update progress copy and remove emoji from runtime labels**

In `handleSubmit`, replace the loading strings and `progressMap` with:

```ts
loadingStatus.value = '正在准备行程请求'

const progressMap: Record<string, { pct: number; label: string }> = {
  search_attractions: { pct: 25, label: '正在检索景点' },
  check_weather: { pct: 50, label: '正在核对天气' },
  find_hotels: { pct: 75, label: '正在匹配住宿' },
  plan_itinerary: { pct: 90, label: '正在编排行程' },
  context_ready: { pct: 82, label: '正在整理上下文' },
  handle_error: { pct: loadingProgress.value, label: '正在恢复流程' },
}
```

After the stream resolves, replace the completion string with:

```ts
loadingStatus.value = '行程已生成'
```

- [ ] **Step 2: Replace the home template**

Use this structure while keeping the existing form bindings, rules, `handleSubmit`, and date picker disabled-date handlers:

```vue
<template>
  <main class="home-page">
    <header class="home-nav">
      <div class="brand-mark">Trip Planner</div>
      <div class="nav-meta">
        <span>景点</span>
        <span>天气</span>
        <span>住宿</span>
        <span>行程</span>
      </div>
    </header>

    <section class="home-workspace">
      <aside class="intro-panel">
        <p class="eyebrow">MULTI-AGENT PLANNING</p>
        <h1>让行程先变得清楚</h1>
        <p class="intro-copy">
          输入目的地、日期和偏好。多 Agent 工作流会结合景点、天气、住宿与路线信息，整理成可审阅的旅行计划。
        </p>
        <div class="trust-grid">
          <div class="trust-item">
            <strong>4</strong>
            <span>协作节点</span>
          </div>
          <div class="trust-item">
            <strong>SSE</strong>
            <span>实时进度</span>
          </div>
          <div class="trust-item">
            <strong>AMap</strong>
            <span>真实地点</span>
          </div>
        </div>
      </aside>

      <a-card class="request-card" :bordered="false">
        <div class="card-heading">
          <div>
            <h2>创建旅行计划</h2>
            <p>规划前先确认核心约束，生成后可继续编辑。</p>
          </div>
          <span class="data-badge">真实数据驱动</span>
        </div>

        <a-form :model="formData" layout="vertical" @finish="handleSubmit">
          <section class="form-section">
            <div class="section-heading">
              <span>目的地与日期</span>
              <small>必填</small>
            </div>
            <a-row :gutter="[16, 12]">
              <a-col :xs="24" :md="9">
                <a-form-item name="city" :rules="[{ required: true, message: '请输入目的地城市' }]">
                  <template #label><span class="form-label">目的地城市</span></template>
                  <a-input v-model:value="formData.city" placeholder="例如：杭州" size="large" class="quiet-input" />
                </a-form-item>
              </a-col>
              <a-col :xs="24" :sm="12" :md="6">
                <a-form-item name="start_date" :rules="[{ required: true, message: '请选择开始日期' }]">
                  <template #label><span class="form-label">开始日期</span></template>
                  <a-date-picker
                    v-model:value="formData.start_date"
                    style="width: 100%"
                    size="large"
                    class="quiet-input"
                    placeholder="选择日期"
                    :disabled-date="disabledStartDate"
                  />
                </a-form-item>
              </a-col>
              <a-col :xs="24" :sm="12" :md="6">
                <a-form-item name="end_date" :rules="[{ required: true, message: '请选择结束日期' }]">
                  <template #label><span class="form-label">结束日期</span></template>
                  <a-date-picker
                    v-model:value="formData.end_date"
                    style="width: 100%"
                    size="large"
                    class="quiet-input"
                    placeholder="选择日期"
                    :disabled-date="disabledEndDate"
                  />
                </a-form-item>
              </a-col>
              <a-col :xs="24" :md="3">
                <a-form-item>
                  <template #label><span class="form-label">天数</span></template>
                  <div class="days-pill">
                    <strong>{{ formData.travel_days }}</strong>
                    <span>天</span>
                  </div>
                </a-form-item>
              </a-col>
            </a-row>
          </section>

          <section class="form-section">
            <div class="section-heading">
              <span>旅行偏好</span>
              <small>用于排序</small>
            </div>
            <a-row :gutter="[16, 12]">
              <a-col :xs="24" :md="8">
                <a-form-item name="transportation">
                  <template #label><span class="form-label">交通方式</span></template>
                  <a-select v-model:value="formData.transportation" size="large" class="quiet-select">
                    <a-select-option value="公共交通">公共交通</a-select-option>
                    <a-select-option value="自驾">自驾</a-select-option>
                    <a-select-option value="步行">步行</a-select-option>
                    <a-select-option value="混合">混合</a-select-option>
                  </a-select>
                </a-form-item>
              </a-col>
              <a-col :xs="24" :md="8">
                <a-form-item name="accommodation">
                  <template #label><span class="form-label">住宿偏好</span></template>
                  <a-select v-model:value="formData.accommodation" size="large" class="quiet-select">
                    <a-select-option value="经济型酒店">经济型酒店</a-select-option>
                    <a-select-option value="舒适型酒店">舒适型酒店</a-select-option>
                    <a-select-option value="豪华酒店">豪华酒店</a-select-option>
                    <a-select-option value="民宿">民宿</a-select-option>
                  </a-select>
                </a-form-item>
              </a-col>
              <a-col :xs="24" :md="8">
                <a-form-item name="preferences">
                  <template #label><span class="form-label">兴趣偏好</span></template>
                  <a-checkbox-group v-model:value="formData.preferences" class="preference-grid">
                    <a-checkbox value="历史文化" class="preference-pill">历史文化</a-checkbox>
                    <a-checkbox value="自然风光" class="preference-pill">自然风光</a-checkbox>
                    <a-checkbox value="美食" class="preference-pill">美食</a-checkbox>
                    <a-checkbox value="购物" class="preference-pill">购物</a-checkbox>
                    <a-checkbox value="艺术" class="preference-pill">艺术</a-checkbox>
                    <a-checkbox value="休闲" class="preference-pill">休闲</a-checkbox>
                  </a-checkbox-group>
                </a-form-item>
              </a-col>
            </a-row>
          </section>

          <section class="form-section">
            <div class="section-heading">
              <span>额外要求</span>
              <small>可选</small>
            </div>
            <a-form-item name="free_text_input">
              <a-textarea
                v-model:value="formData.free_text_input"
                placeholder="例如：需要无障碍设施、避开夜间步行、对海鲜过敏"
                :rows="3"
                size="large"
                class="quiet-textarea"
              />
            </a-form-item>
          </section>

          <a-form-item class="submit-row">
            <a-button type="primary" html-type="submit" :loading="loading" size="large" block class="submit-button">
              <span>{{ loading ? '正在生成' : '生成行程' }}</span>
            </a-button>
          </a-form-item>

          <a-form-item v-if="loading">
            <div class="loading-panel">
              <div class="loading-copy">
                <span>生成进度</span>
                <strong>{{ loadingStatus }}</strong>
              </div>
              <a-progress :percent="loadingProgress" status="active" stroke-color="#0f766e" :stroke-width="8" />
            </div>
          </a-form-item>
        </a-form>
      </a-card>
    </section>
  </main>
</template>
```

- [ ] **Step 3: Replace the home scoped styles**

Replace the entire `<style scoped>` content in `Home.vue` with CSS implementing the approved palette:

```css
.home-page {
  min-height: 100vh;
  padding: 28px clamp(16px, 3vw, 40px) 48px;
  background:
    radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
    #f6f8f7;
  color: #1f2a28;
}

.home-nav {
  max-width: 1180px;
  margin: 0 auto 44px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.brand-mark {
  font-size: 16px;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.nav-meta {
  display: flex;
  gap: 18px;
  color: #66736f;
  font-size: 13px;
}

.home-workspace {
  max-width: 1180px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(280px, 0.82fr) minmax(0, 1.18fr);
  gap: clamp(24px, 4vw, 56px);
  align-items: start;
}

.intro-panel {
  padding-top: 28px;
}

.eyebrow {
  margin: 0 0 12px;
  color: #0f766e;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.intro-panel h1 {
  margin: 0;
  max-width: 460px;
  font-size: clamp(36px, 5vw, 64px);
  line-height: 1.04;
  font-weight: 850;
  letter-spacing: 0;
}

.intro-copy {
  max-width: 520px;
  margin: 20px 0 26px;
  color: #66736f;
  font-size: 16px;
  line-height: 1.75;
}

.trust-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.trust-item {
  padding: 14px;
  background: #ffffff;
  border: 1px solid #dfe5e2;
  border-radius: 8px;
}

.trust-item strong {
  display: block;
  color: #0f766e;
  font-size: 20px;
}

.trust-item span {
  color: #66736f;
  font-size: 12px;
}

.request-card {
  border: 1px solid #dfe5e2;
  border-radius: 8px;
  box-shadow: 0 18px 44px rgba(31, 42, 40, 0.08);
}

.card-heading {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 24px;
}

.card-heading h2 {
  margin: 0 0 6px;
  font-size: 22px;
  color: #1f2a28;
}

.card-heading p {
  margin: 0;
  color: #66736f;
}

.data-badge {
  flex: 0 0 auto;
  padding: 6px 10px;
  border-radius: 999px;
  color: #0f766e;
  background: #e8f5f2;
  border: 1px solid rgba(15, 118, 110, 0.18);
  font-size: 12px;
  font-weight: 700;
}

.form-section {
  padding: 18px 0 6px;
  border-top: 1px solid #edf2f0;
}

.form-section:first-of-type {
  border-top: 0;
  padding-top: 0;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.section-heading span {
  color: #1f2a28;
  font-size: 15px;
  font-weight: 800;
}

.section-heading small {
  color: #8a9692;
}

.form-label {
  color: #46524f;
  font-weight: 650;
}

.quiet-input :deep(.ant-input),
.quiet-input :deep(.ant-picker),
.quiet-textarea :deep(.ant-input),
.quiet-select :deep(.ant-select-selector) {
  border-color: #d8e0dd !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

.quiet-input :deep(.ant-input:hover),
.quiet-input :deep(.ant-picker:hover),
.quiet-textarea :deep(.ant-input:hover),
.quiet-select:hover :deep(.ant-select-selector) {
  border-color: #0f766e !important;
}

.quiet-input :deep(.ant-input:focus),
.quiet-input :deep(.ant-picker-focused),
.quiet-textarea :deep(.ant-input:focus),
.quiet-select :deep(.ant-select-focused .ant-select-selector) {
  border-color: #0f766e !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.1) !important;
}

.days-pill {
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border-radius: 6px;
  color: #0f766e;
  background: #e8f5f2;
  border: 1px solid rgba(15, 118, 110, 0.18);
}

.days-pill strong {
  font-size: 22px;
}

.preference-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.preference-pill {
  margin: 0;
}

.preference-pill :deep(.ant-checkbox) {
  display: none;
}

.preference-pill :deep(span:last-child) {
  display: inline-flex;
  padding: 7px 12px;
  border: 1px solid #d8e0dd;
  border-radius: 999px;
  color: #46524f;
  background: #ffffff;
  transition: all 0.18s ease;
}

.preference-pill :deep(.ant-checkbox-checked + span) {
  color: #0f766e;
  background: #e8f5f2;
  border-color: rgba(15, 118, 110, 0.32);
}

.submit-row {
  margin-top: 20px;
}

.submit-button {
  height: 46px;
  border-radius: 6px;
  background: #0f766e;
  border-color: #0f766e;
  font-weight: 800;
  box-shadow: none;
}

.submit-button:hover,
.submit-button:focus {
  background: #0b5f59 !important;
  border-color: #0b5f59 !important;
}

.loading-panel {
  padding: 14px;
  background: #f8fbfa;
  border: 1px solid #dfe5e2;
  border-radius: 8px;
}

.loading-copy {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
  color: #66736f;
}

.loading-copy strong {
  color: #0f766e;
}

@media (max-width: 900px) {
  .home-workspace {
    grid-template-columns: 1fr;
  }

  .intro-panel {
    padding-top: 0;
  }
}

@media (max-width: 640px) {
  .home-page {
    padding: 20px 14px 32px;
  }

  .home-nav {
    margin-bottom: 28px;
    align-items: flex-start;
    flex-direction: column;
  }

  .nav-meta {
    flex-wrap: wrap;
    gap: 10px;
  }

  .trust-grid {
    grid-template-columns: 1fr;
  }

  .card-heading {
    flex-direction: column;
  }
}
```

- [ ] **Step 4: Verify home page build**

Run from `frontend`:

```bash
npm run build
```

Expected: build passes.

- [ ] **Step 5: Commit home refresh**

```bash
git add frontend/src/views/Home.vue
git commit -m "style: refresh trip request page"
```

---

### Task 3: Result Page Structure And Summary Helpers

**Files:**
- Modify: `frontend/src/views/Result.vue`

- [ ] **Step 1: Add computed import**

Change the Vue import to:

```ts
import { computed, ref, onMounted, nextTick } from 'vue'
```

- [ ] **Step 2: Add summary helpers after map state**

Add these computed values below `let map: any = null`:

```ts
const totalAttractions = computed(() => (
  tripPlan.value?.days.reduce((sum, day) => sum + day.attractions.length, 0) ?? 0
))

const totalHotels = computed(() => (
  tripPlan.value?.days.filter(day => !!day.hotel).length ?? 0
))

const weatherCount = computed(() => tripPlan.value?.weather_info?.length ?? 0)

const dateRangeLabel = computed(() => {
  if (!tripPlan.value) return ''
  return `${tripPlan.value.start_date} 至 ${tripPlan.value.end_date}`
})
```

- [ ] **Step 3: Replace the top-level result template structure**

Keep all existing event handlers and loops, but replace the visible structure with:

```vue
<template>
  <main class="result-page">
    <section v-if="tripPlan" class="result-shell">
      <header class="result-header">
        <div>
          <a-button class="text-button" type="link" @click="goBack">返回首页</a-button>
          <h1>{{ tripPlan.city }}行程</h1>
          <p>{{ dateRangeLabel }}</p>
        </div>
        <a-space size="small" wrap>
          <a-button v-if="!editMode" @click="toggleEditMode">编辑行程</a-button>
          <a-button v-else type="primary" @click="saveChanges">保存修改</a-button>
          <a-button v-if="editMode" @click="cancelEdit">取消编辑</a-button>
          <a-dropdown v-if="!editMode">
            <template #overlay>
              <a-menu>
                <a-menu-item key="image" @click="exportAsImage">导出图片</a-menu-item>
                <a-menu-item key="pdf" @click="exportAsPDF">导出 PDF</a-menu-item>
              </a-menu>
            </template>
            <a-button>导出行程 <DownOutlined /></a-button>
          </a-dropdown>
        </a-space>
      </header>

      <div class="summary-grid">
        <div class="metric-card">
          <span>行程天数</span>
          <strong>{{ tripPlan.days.length }}</strong>
        </div>
        <div class="metric-card">
          <span>景点数量</span>
          <strong>{{ totalAttractions }}</strong>
        </div>
        <div class="metric-card">
          <span>住宿匹配</span>
          <strong>{{ totalHotels }}</strong>
        </div>
        <div class="metric-card accent">
          <span>天气记录</span>
          <strong>{{ weatherCount }}</strong>
        </div>
      </div>

      <div class="content-wrapper">
        <aside class="side-nav">
          <a-affix :offset-top="24">
            <a-menu mode="inline" :selected-keys="[activeSection]" @click="scrollToSection">
              <a-menu-item key="overview">概览</a-menu-item>
              <a-menu-item key="budget" v-if="tripPlan.budget">预算</a-menu-item>
              <a-menu-item key="map">地图</a-menu-item>
              <a-sub-menu key="days" title="每日行程">
                <a-menu-item v-for="(day, index) in tripPlan.days" :key="`day-${index}`">
                  第{{ day.day_index + 1 }}天
                </a-menu-item>
              </a-sub-menu>
              <a-menu-item key="weather" v-if="tripPlan.weather_info && tripPlan.weather_info.length > 0">
                天气
              </a-menu-item>
            </a-menu>
          </a-affix>
        </aside>

        <section class="main-content">
          <!-- Existing overview, budget, map, day, hotel, meal, and weather content is rebuilt in Step 4. -->
        </section>
      </div>
    </section>

    <a-empty v-else class="empty-state" description="没有找到旅行计划数据">
      <template #description>
        <span>暂无旅行计划数据，请先创建行程。</span>
      </template>
      <a-button type="primary" @click="goBack">返回首页创建行程</a-button>
    </a-empty>

    <a-back-top :visibility-height="300">
      <div class="back-top-button">↑</div>
    </a-back-top>
  </main>
</template>
```

- [ ] **Step 4: Reinsert main result content inside `.main-content`**

Inside `<section class="main-content">`, add:

```vue
<section id="overview" class="panel overview-panel">
  <div class="panel-heading">
    <h2>行程概览</h2>
    <span>生成结果</span>
  </div>
  <p class="suggestion-text">{{ tripPlan.overall_suggestions }}</p>
</section>

<section id="budget" v-if="tripPlan.budget" class="panel budget-panel">
  <div class="panel-heading">
    <h2>预算</h2>
    <span>预估费用</span>
  </div>
  <div class="budget-grid">
    <div class="budget-item">
      <span>景点门票</span>
      <strong>¥{{ tripPlan.budget.total_attractions }}</strong>
    </div>
    <div class="budget-item">
      <span>酒店住宿</span>
      <strong>¥{{ tripPlan.budget.total_hotels }}</strong>
    </div>
    <div class="budget-item">
      <span>餐饮费用</span>
      <strong>¥{{ tripPlan.budget.total_meals }}</strong>
    </div>
    <div class="budget-item">
      <span>交通费用</span>
      <strong>¥{{ tripPlan.budget.total_transportation }}</strong>
    </div>
  </div>
  <div class="budget-total">
    <span>预估总费用</span>
    <strong>¥{{ tripPlan.budget.total }}</strong>
  </div>
</section>

<section id="map" class="panel map-panel">
  <div class="panel-heading">
    <h2>地图</h2>
    <span>景点位置与路线</span>
  </div>
  <div id="amap-container"></div>
</section>

<section class="panel days-panel">
  <div class="panel-heading">
    <h2>每日行程</h2>
    <span>可编辑</span>
  </div>
  <a-collapse v-model:activeKey="activeDays" accordion ghost>
    <a-collapse-panel v-for="(day, index) in tripPlan.days" :key="index" :id="`day-${index}`">
      <template #header>
        <div class="day-header">
          <span>第{{ day.day_index + 1 }}天</span>
          <small>{{ day.date }}</small>
        </div>
      </template>

      <div class="day-context">
        <p>{{ day.description }}</p>
        <div class="context-list">
          <span>交通：{{ day.transportation }}</span>
          <span>住宿：{{ day.accommodation }}</span>
        </div>
      </div>

      <div class="subsection-title">景点安排</div>
      <div class="attraction-list">
        <article v-for="(item, attrIndex) in day.attractions" :key="`${day.day_index}-${item.name}-${attrIndex}`" class="attraction-row">
          <div class="attraction-media">
            <img :src="getAttractionImage(item.name, attrIndex)" :alt="item.name" @error="handleImageError" />
            <span>{{ attrIndex + 1 }}</span>
          </div>
          <div class="attraction-body">
            <div class="attraction-title-row">
              <h3>{{ item.name }}</h3>
              <span v-if="item.ticket_price" class="price-tag">¥{{ item.ticket_price }}</span>
            </div>

            <div v-if="editMode" class="edit-fields">
              <a-input v-model:value="item.address" size="small" />
              <a-input-number v-model:value="item.visit_duration" :min="10" :max="480" size="small" />
              <a-textarea v-model:value="item.description" :rows="2" size="small" />
              <a-space>
                <a-button size="small" @click="moveAttraction(day.day_index, attrIndex, 'up')" :disabled="attrIndex === 0">上移</a-button>
                <a-button size="small" @click="moveAttraction(day.day_index, attrIndex, 'down')" :disabled="attrIndex === day.attractions.length - 1">下移</a-button>
                <a-button size="small" danger @click="deleteAttraction(day.day_index, attrIndex)">删除</a-button>
              </a-space>
            </div>

            <div v-else class="attraction-meta">
              <p>{{ item.description }}</p>
              <span>{{ item.address }}</span>
              <span>{{ item.visit_duration }} 分钟</span>
              <span v-if="item.rating">评分 {{ item.rating }}</span>
            </div>
          </div>
        </article>
      </div>

      <div v-if="day.hotel" class="hotel-panel">
        <div class="subsection-title">住宿推荐</div>
        <a-descriptions :column="2" size="small" bordered>
          <a-descriptions-item label="名称">{{ day.hotel.name }}</a-descriptions-item>
          <a-descriptions-item label="类型">{{ day.hotel.type }}</a-descriptions-item>
          <a-descriptions-item label="地址">{{ day.hotel.address }}</a-descriptions-item>
          <a-descriptions-item label="价格">{{ day.hotel.price_range }}</a-descriptions-item>
          <a-descriptions-item label="评分">{{ day.hotel.rating || '暂无' }}</a-descriptions-item>
          <a-descriptions-item label="距离">{{ day.hotel.distance }}</a-descriptions-item>
        </a-descriptions>
      </div>

      <div class="meal-panel">
        <div class="subsection-title">餐饮安排</div>
        <a-descriptions :column="1" bordered size="small">
          <a-descriptions-item v-for="meal in day.meals" :key="meal.type" :label="getMealLabel(meal.type)">
            {{ meal.name }}<span v-if="meal.description"> - {{ meal.description }}</span>
          </a-descriptions-item>
        </a-descriptions>
      </div>
    </a-collapse-panel>
  </a-collapse>
</section>

<section id="weather" v-if="tripPlan.weather_info && tripPlan.weather_info.length > 0" class="panel weather-panel">
  <div class="panel-heading">
    <h2>天气</h2>
    <span>按日期匹配</span>
  </div>
  <div class="weather-grid">
    <article v-for="item in tripPlan.weather_info" :key="item.date" class="weather-card">
      <strong>{{ item.date }}</strong>
      <div>
        <span>白天</span>
        <b>{{ item.day_weather }} {{ item.day_temp }}°C</b>
      </div>
      <div>
        <span>夜间</span>
        <b>{{ item.night_weather }} {{ item.night_temp }}°C</b>
      </div>
      <small>{{ item.wind_direction }} {{ item.wind_power }}</small>
    </article>
  </div>
</section>
```

- [ ] **Step 5: Verify result structure compiles**

Run from `frontend`:

```bash
npm run build
```

Expected: build passes.

- [ ] **Step 6: Commit result structure**

```bash
git add frontend/src/views/Result.vue
git commit -m "style: restructure itinerary result page"
```

---

### Task 4: Result Page Visuals, Map, Image Fallbacks, And Export Styles

**Files:**
- Modify: `frontend/src/views/Result.vue`

- [ ] **Step 1: Replace image fallback generation**

Replace `getAttractionImage()` fallback with a neutral SVG:

```ts
const getAttractionImage = (name: string, index: number): string => {
  if (attractionPhotos.value[name]) {
    return attractionPhotos.value[name]
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260">
    <rect width="400" height="260" fill="#eef2f0"/>
    <circle cx="62" cy="62" r="24" fill="#d8e0dd"/>
    <path d="M0 198 L112 118 L206 176 L286 104 L400 184 L400 260 L0 260 Z" fill="#d8e0dd"/>
    <text x="28" y="232" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#66736f">${name || `景点 ${index + 1}`}</text>
  </svg>`

  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}
```

Replace `handleImageError()` with:

```ts
const handleImageError = (event: Event) => {
  const img = event.target as HTMLImageElement
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260"><rect width="400" height="260" fill="#eef2f0"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#66736f">图片暂不可用</text></svg>'
  img.src = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}
```

- [ ] **Step 2: Update map marker and route colors**

In `addAttractionMarkers()`, replace marker label content with:

```ts
label: {
  content: `<div style="background:#0f766e;color:#fff;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:700;box-shadow:0 4px 10px rgba(15,118,110,.24);">${index + 1}</div>`,
  offset: new AMap.Pixel(0, -30)
}
```

In `InfoWindow` content, replace the blue inline color with teal:

```html
<p style="margin: 4px 0; color: #0f766e;"><strong>第${attraction.dayIndex + 1}天 景点${attraction.attrIndex + 1}</strong></p>
```

In `drawRoutes()`, set:

```ts
strokeColor: '#0f766e',
strokeWeight: 4,
strokeOpacity: 0.78,
```

- [ ] **Step 3: Replace export styles**

Replace `applyExportStyles()` with:

```ts
const applyExportStyles = (container: HTMLElement) => {
  container.querySelectorAll('.panel, .ant-card').forEach((card) => {
    const el = card as HTMLElement
    el.style.cssText = 'background-color:#fff;border:1px solid #dfe5e2;border-radius:8px;box-shadow:none;margin-bottom:16px;overflow:hidden'
  })
  container.querySelectorAll('.panel-heading, .ant-card-head').forEach((head) => {
    const el = head as HTMLElement
    el.style.cssText = 'background-color:#fff;color:#1f2a28;border-bottom:1px solid #dfe5e2;padding:14px 18px;font-size:16px;font-weight:700'
  })
  container.querySelectorAll('.ant-card-body').forEach((body) => {
    const el = body as HTMLElement
    el.style.cssText = 'background-color:#fff;padding:18px'
  })
  container.querySelectorAll('.budget-total').forEach((item) => {
    const el = item as HTMLElement
    el.style.cssText = 'background-color:#e8f5f2;color:#0f766e;padding:14px;border-radius:8px;margin-top:12px'
  })
  container.querySelectorAll('.price-tag').forEach((tag) => {
    const el = tag as HTMLElement
    el.style.cssText = 'background-color:#fff7f4;color:#f0765f;border:1px solid #ffd3c4;border-radius:999px;padding:3px 8px;font-weight:700'
  })
}
```

In `prepareExportContainer()`, change background colors to:

```ts
container.style.backgroundColor = '#f6f8f7'
```

and:

```ts
backgroundColor: '#f6f8f7',
```

- [ ] **Step 4: Replace the result scoped styles**

Replace the entire `<style scoped>` block in `Result.vue` with:

```css
.result-page {
  min-height: 100vh;
  padding: 28px clamp(14px, 3vw, 40px) 48px;
  background: #f6f8f7;
  color: #1f2a28;
}

.result-shell {
  max-width: 1380px;
  margin: 0 auto;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  margin-bottom: 18px;
}

.result-header h1 {
  margin: 4px 0 6px;
  color: #1f2a28;
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1.08;
  font-weight: 850;
  letter-spacing: 0;
}

.result-header p {
  margin: 0;
  color: #66736f;
}

.text-button {
  padding: 0;
  color: #0f766e;
  font-weight: 700;
}

.result-page :deep(.ant-btn-primary) {
  background: #0f766e;
  border-color: #0f766e;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.metric-card {
  padding: 16px;
  background: #ffffff;
  border: 1px solid #dfe5e2;
  border-radius: 8px;
}

.metric-card span {
  display: block;
  color: #66736f;
  font-size: 13px;
}

.metric-card strong {
  display: block;
  margin-top: 8px;
  color: #1f2a28;
  font-size: 28px;
}

.metric-card.accent {
  background: #fff7f4;
  border-color: #ffd3c4;
}

.content-wrapper {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}

.side-nav {
  min-width: 0;
}

.side-nav :deep(.ant-menu) {
  border: 1px solid #dfe5e2;
  border-radius: 8px;
  background: #ffffff;
}

.side-nav :deep(.ant-menu-item),
.side-nav :deep(.ant-menu-submenu-title) {
  border-radius: 6px;
  color: #46524f;
}

.side-nav :deep(.ant-menu-item-selected) {
  color: #0f766e;
  background: #e8f5f2;
}

.main-content {
  min-width: 0;
}

.panel {
  margin-bottom: 16px;
  background: #ffffff;
  border: 1px solid #dfe5e2;
  border-radius: 8px;
  overflow: hidden;
}

.panel-heading {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 16px 18px;
  border-bottom: 1px solid #edf2f0;
}

.panel-heading h2 {
  margin: 0;
  color: #1f2a28;
  font-size: 18px;
}

.panel-heading span {
  color: #66736f;
  font-size: 12px;
}

.suggestion-text {
  margin: 0;
  padding: 18px;
  color: #46524f;
  line-height: 1.75;
}

.budget-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  padding: 18px;
}

.budget-item {
  padding: 14px;
  background: #f8fbfa;
  border: 1px solid #edf2f0;
  border-radius: 8px;
}

.budget-item span,
.budget-total span {
  display: block;
  color: #66736f;
  font-size: 13px;
}

.budget-item strong,
.budget-total strong {
  display: block;
  margin-top: 8px;
  color: #1f2a28;
  font-size: 22px;
}

.budget-total {
  margin: 0 18px 18px;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #e8f5f2;
  border-radius: 8px;
}

.budget-total strong {
  margin: 0;
  color: #0f766e;
  font-size: 28px;
}

.map-panel {
  min-height: 420px;
}

#amap-container {
  width: 100%;
  height: 420px;
}

.days-panel {
  padding-bottom: 2px;
}

.days-panel :deep(.ant-collapse) {
  background: transparent;
}

.days-panel :deep(.ant-collapse-item) {
  border-color: #edf2f0;
}

.days-panel :deep(.ant-collapse-header) {
  padding: 16px 18px !important;
}

.days-panel :deep(.ant-collapse-content-box) {
  padding: 0 18px 18px;
}

.day-header {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.day-header span {
  color: #1f2a28;
  font-weight: 800;
}

.day-header small {
  color: #66736f;
}

.day-context {
  padding: 14px;
  background: #f8fbfa;
  border: 1px solid #edf2f0;
  border-radius: 8px;
  margin-bottom: 18px;
}

.day-context p {
  margin: 0 0 10px;
  color: #46524f;
  line-height: 1.7;
}

.context-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.context-list span {
  padding: 5px 9px;
  color: #0f766e;
  background: #e8f5f2;
  border-radius: 999px;
  font-size: 12px;
}

.subsection-title {
  margin: 18px 0 10px;
  color: #1f2a28;
  font-weight: 800;
}

.attraction-list {
  display: grid;
  gap: 12px;
}

.attraction-row {
  display: grid;
  grid-template-columns: 158px minmax(0, 1fr);
  gap: 14px;
  padding: 12px;
  border: 1px solid #edf2f0;
  border-radius: 8px;
  background: #ffffff;
}

.attraction-media {
  position: relative;
  overflow: hidden;
  border-radius: 8px;
  background: #eef2f0;
  aspect-ratio: 4 / 3;
}

.attraction-media img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.attraction-media span {
  position: absolute;
  top: 10px;
  left: 10px;
  min-width: 28px;
  height: 28px;
  padding: 0 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #ffffff;
  background: #0f766e;
  font-weight: 800;
}

.attraction-body {
  min-width: 0;
}

.attraction-title-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.attraction-title-row h3 {
  margin: 0;
  color: #1f2a28;
  font-size: 17px;
}

.price-tag {
  flex: 0 0 auto;
  padding: 3px 8px;
  color: #f0765f;
  background: #fff7f4;
  border: 1px solid #ffd3c4;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
}

.attraction-meta p {
  margin: 8px 0 10px;
  color: #46524f;
  line-height: 1.65;
}

.attraction-meta span {
  display: inline-flex;
  margin: 0 8px 6px 0;
  color: #66736f;
  font-size: 12px;
}

.edit-fields {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}

.hotel-panel,
.meal-panel {
  margin-top: 16px;
}

.weather-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  padding: 18px;
}

.weather-card {
  padding: 14px;
  border: 1px solid #dfe5e2;
  border-radius: 8px;
  background: #f8fbfa;
}

.weather-card strong {
  color: #1f2a28;
}

.weather-card div {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 10px;
}

.weather-card span,
.weather-card small {
  color: #66736f;
}

.weather-card b {
  color: #0f766e;
}

.empty-state {
  min-height: 80vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.back-top-button {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  background: #0f766e;
  border-radius: 999px;
  font-weight: 800;
  box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22);
}

@media (max-width: 1100px) {
  .content-wrapper {
    grid-template-columns: 1fr;
  }

  .side-nav {
    display: none;
  }
}

@media (max-width: 780px) {
  .result-header {
    flex-direction: column;
  }

  .summary-grid,
  .budget-grid,
  .weather-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .attraction-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 520px) {
  .result-page {
    padding: 20px 12px 34px;
  }

  .summary-grid,
  .budget-grid,
  .weather-grid {
    grid-template-columns: 1fr;
  }

  .budget-total {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }
}
```

- [ ] **Step 5: Verify result visual build**

Run from `frontend`:

```bash
npm run build
```

Expected: build passes.

- [ ] **Step 6: Commit result visual refresh**

```bash
git add frontend/src/views/Result.vue
git commit -m "style: refresh itinerary visual system"
```

---

### Task 5: Browser QA And Final Verification

**Files:**
- Modify only files with defects found during QA.

- [ ] **Step 1: Run production build**

Run from `frontend`:

```bash
npm run build
```

Expected: build passes with no TypeScript errors.

- [ ] **Step 2: Start local dev server**

Run from `frontend`:

```bash
npm run dev
```

Expected: Vite serves the frontend at the printed localhost URL, usually `http://localhost:5173/`.

- [ ] **Step 3: Check home page manually**

Open `/` in the browser and verify:

- The page no longer shows purple gradients, floating circles, bouncing icons, or emoji-heavy labels.
- The form remains usable on desktop and mobile widths.
- Date validation still blocks past start dates and invalid end dates.
- Loading progress text is plain and uses teal progress styling.

- [ ] **Step 4: Check result page manually with sessionStorage sample**

If no generated plan is available, create sessionStorage data from the current `TripPlan` type shape in the browser console. Use real-looking local sample data only for manual QA and do not commit it.

Verify `/result`:

- Header summary renders city, date range, days, attractions, hotels, and weather count.
- Edit mode still edits address, duration, description, and attraction order.
- Save writes back to sessionStorage.
- Cancel restores the original plan.
- Map panel keeps its fixed height and does not collapse when AMap key is missing.
- Weather and budget panels wrap without overflow.

- [ ] **Step 5: Check export controls**

Click export image and export PDF on `/result`.

Expected:

- Export starts without JavaScript errors.
- Exported content uses the refreshed white/teal visual system instead of old purple card headers.
- If the map canvas cannot be captured, the rest of the export still completes as before.

- [ ] **Step 6: Final repository verification**

Run:

```bash
git status --short --branch
```

Expected: only intentional frontend files are modified before final commit.

Run from `frontend`:

```bash
npm run build
```

Expected: build passes.

- [ ] **Step 7: Final commit**

```bash
git add frontend/src/App.vue frontend/src/views/Home.vue frontend/src/views/Result.vue
git commit -m "style: refresh travel planner UI"
```

---

## Self-Review

- Spec coverage: The plan covers the app shell, home page, result page, Graphite / Teal / Coral palette, reduced AI wording, removal of emoji-heavy labels, map styling, image fallback styling, export styling, and responsive checks.
- Scope control: The plan does not change backend behavior, API schemas, routing, or dependencies.
- Type consistency: New computed helpers use existing `TripPlan`, `DayPlan`, and template properties already defined in `frontend/src/types/index.ts`.
- Verification: Each implementation phase includes `npm run build`; final QA includes browser checks for home, result, edit mode, map, and export.

<template>
  <main class="home-page">
    <header class="home-nav">
      <div class="brand-mark">Trip Planner</div>
      <div class="nav-meta" aria-label="页面导航">
        <a href="#inspiration">灵感</a>
        <a href="#plan-form">规划</a>
        <a href="#workflow">流程</a>
        <a href="/inspiration/ATTRIBUTIONS.md" target="_blank" rel="noopener">图片来源</a>
      </div>
    </header>

    <section class="home-workspace">
      <aside
        class="destination-feature"
        :class="{ 'is-image-fallback': !featuredPreset.imageAvailable || failedPresetImages[featuredPreset.id] }"
      >
        <img
          v-if="featuredPreset.imageAvailable && !failedPresetImages[featuredPreset.id]"
          :src="featuredPreset.imageSrc"
          :srcset="featuredPreset.imageSrcSet"
          sizes="(min-width: 1520px) 577px, (min-width: 1283px) calc(41vw - 46px), (min-width: 1231px) 480px, (min-width: 641px) 94vw, calc(100vw - 28px)"
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
          <span class="destination-feature__duration">{{ featuredPreset.recommendedDays }} 天推荐停留</span>
        </div>
      </aside>

      <a-card id="plan-form" class="request-card" :bordered="false">
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
                  <template #label>
                    <span class="form-label">目的地城市</span>
                  </template>
                  <a-input v-model:value="formData.city" placeholder="例如：杭州" size="large" class="quiet-input" />
                </a-form-item>
              </a-col>
              <a-col :xs="24" :sm="12" :md="6">
                <a-form-item name="start_date" :rules="[{ required: true, message: '请选择开始日期' }]">
                  <template #label>
                    <span class="form-label">开始日期</span>
                  </template>
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
                  <template #label>
                    <span class="form-label">结束日期</span>
                  </template>
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
                  <template #label>
                    <span class="form-label">天数</span>
                  </template>
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
              <a-col :xs="24" :md="6">
                <a-form-item name="transportation">
                  <template #label>
                    <span class="form-label">交通方式</span>
                  </template>
                  <a-select v-model:value="formData.transportation" size="large" class="quiet-select">
                    <a-select-option value="公共交通">公共交通</a-select-option>
                    <a-select-option value="自驾">自驾</a-select-option>
                    <a-select-option value="步行">步行</a-select-option>
                    <a-select-option value="混合">混合</a-select-option>
                  </a-select>
                </a-form-item>
              </a-col>
              <a-col :xs="24" :md="6">
                <a-form-item name="accommodation">
                  <template #label>
                    <span class="form-label">住宿偏好</span>
                  </template>
                  <a-select v-model:value="formData.accommodation" size="large" class="quiet-select">
                    <a-select-option value="经济型酒店">经济型酒店</a-select-option>
                    <a-select-option value="舒适型酒店">舒适型酒店</a-select-option>
                    <a-select-option value="豪华酒店">豪华酒店</a-select-option>
                    <a-select-option value="民宿">民宿</a-select-option>
                  </a-select>
                </a-form-item>
              </a-col>
              <a-col :xs="24" :md="12">
                <a-form-item name="preferences">
                  <template #label>
                    <span class="form-label">兴趣偏好</span>
                  </template>
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
                :rows="2"
                size="large"
                class="quiet-textarea"
              />
            </a-form-item>
          </section>

          <div v-if="planSummaryItems.length" class="plan-summary" role="group" aria-label="当前计划">
            <span class="plan-summary__label">当前计划</span>
            <span v-for="item in planSummaryItems" :key="item" class="plan-summary__item">{{ item }}</span>
          </div>

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
              <a-progress :percent="loadingProgress" status="active" stroke-color="#165dff" :stroke-width="8" />
            </div>
          </a-form-item>
        </a-form>
      </a-card>
    </section>

    <section id="inspiration" class="inspiration-section" aria-labelledby="inspiration-title">
      <div class="section-intro">
        <p class="eyebrow">目的地灵感</p>
        <h2 id="inspiration-title">从一个目的地开始</h2>
        <p>选择喜欢的城市节奏，再按自己的时间调整细节。</p>
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
          <span class="preset-tile__media">
            <img
              v-if="preset.imageAvailable && !failedPresetImages[preset.id]"
              :src="preset.imageSrc"
              :srcset="preset.imageSrcSet"
              sizes="(min-width: 1520px) 348px, (min-width: 1334px) calc(25vw - 32px), (min-width: 901px) calc(23.5vw - 12px), min(78vw, 300px)"
              :alt="preset.imageAlt"
              loading="lazy"
              @error="markPresetImageFailed(preset.id)"
            />
            <span v-else class="preset-tile__placeholder">{{ preset.city }}</span>
            <span class="preset-tile__scrim" aria-hidden="true"></span>
            <span class="preset-tile__meta">{{ preset.city }} · {{ preset.recommendedDays }} 天</span>
          </span>
          <span class="preset-tile__content">
            <strong>{{ preset.title }}</strong>
            <span class="preset-tile__description">{{ preset.description }}</span>
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
  </main>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { tripPresets, type TripPreset } from '@/data/tripPresets'
import { generateTripPlanStream } from '@/services/api'
import type { TripFormData } from '@/types'
import { suggestPresetEndDate } from '@/utils/presetDates'
import { reconcileSelectedPresetId } from '@/utils/presetSelection'
import { getPlanSummaryItems } from '@/utils/planSummary'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

type TripFormState = Omit<TripFormData, 'start_date' | 'end_date'> & {
  start_date: Dayjs | null
  end_date: Dayjs | null
}

const disabledStartDate = (current: Dayjs) => {
  return !!current && current.isBefore(dayjs().startOf('day'))
}

const disabledEndDate = (current: Dayjs) => {
  if (!formData.start_date) {
    return !!current && current.isBefore(dayjs().startOf('day'))
  }
  return !!current && current.isBefore(formData.start_date.startOf('day'))
}

const router = useRouter()
const loading = ref(false)
const loadingProgress = ref(0)
const loadingStatus = ref('')

const formData = reactive<TripFormState>({
  city: '',
  start_date: null,
  end_date: null,
  travel_days: 0,
  transportation: '公共交通',
  accommodation: '经济型酒店',
  preferences: [],
  free_text_input: ''
})

const selectedPresetId = ref<TripPreset['id'] | null>(null)
const failedPresetImages = reactive<Partial<Record<TripPreset['id'], true>>>({})
const featuredPreset = tripPresets[0]!
const selectedPreset = computed(
  () => tripPresets.find((preset) => preset.id === selectedPresetId.value) ?? null,
)
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

watch(() => formData.start_date, (start) => {
  if (start && !formData.end_date && selectedPreset.value) {
    const suggestedEndDate = suggestPresetEndDate(start, formData.end_date, selectedPreset.value.recommendedDays)
    if (suggestedEndDate) {
      formData.end_date = suggestedEndDate
    }
  }
})

watch(() => formData.city, (city) => {
  selectedPresetId.value = reconcileSelectedPresetId(selectedPresetId.value, city, tripPresets)
})

watch([() => formData.start_date, () => formData.end_date], ([start, end]) => {
  if (!start || !end) {
    formData.travel_days = 0
    return
  }

  const days = end.diff(start, 'day') + 1
  if (days > 0 && days <= 30) {
    formData.travel_days = days
  } else if (days > 30) {
    formData.travel_days = 0
    message.warning('旅行天数不能超过30天')
    formData.end_date = null
  } else {
    formData.travel_days = 0
    message.warning('结束日期不能早于开始日期')
    formData.end_date = null
  }
})

const handleSubmit = async () => {
  if (!formData.start_date || !formData.end_date) {
    message.error('请选择日期')
    return
  }

  loading.value = true
  loadingProgress.value = 0
  loadingStatus.value = '正在准备行程请求'

  const progressMap: Record<string, { pct: number; label: string }> = {
    search_attractions: { pct: 25, label: '正在检索景点' },
    check_weather: { pct: 50, label: '正在核对天气' },
    find_hotels: { pct: 75, label: '正在匹配住宿' },
    context_ready: { pct: 82, label: '正在整理上下文' },
    plan_itinerary: { pct: 90, label: '正在编排行程' },
    handle_error: { pct: loadingProgress.value, label: '正在恢复流程' },
  }

  try {
    const requestData: TripFormData = {
      city: formData.city,
      start_date: formData.start_date.format('YYYY-MM-DD'),
      end_date: formData.end_date.format('YYYY-MM-DD'),
      travel_days: formData.travel_days,
      transportation: formData.transportation,
      accommodation: formData.accommodation,
      preferences: formData.preferences,
      free_text_input: formData.free_text_input
    }

    const response = await generateTripPlanStream(requestData, (step, msg) => {
      const info = progressMap[step]
      if (info) {
        loadingProgress.value = info.pct
        loadingStatus.value = info.label
      } else {
        loadingStatus.value = msg
      }
    })

    loadingProgress.value = 100
    loadingStatus.value = '行程已生成'

    if (response.success && response.data) {
      sessionStorage.setItem('tripPlan', JSON.stringify(response.data))
      message.success('旅行计划生成成功')
      setTimeout(() => { router.push('/result') }, 500)
    } else {
      message.error(response.message || '生成失败')
    }
  } catch (error: any) {
    message.error(error.message || '生成旅行计划失败，请稍后重试')
  } finally {
    setTimeout(() => {
      loading.value = false
      loadingProgress.value = 0
      loadingStatus.value = ''
    }, 1000)
  }
}
</script>

<style scoped>
.home-page {
  min-height: 100vh;
  padding: 28px clamp(16px, 3vw, 40px) 48px;
  overflow-x: clip;
  background: #ffffff;
  color: #111111;
}

.home-page,
.home-page * {
  box-sizing: border-box;
}

.home-nav {
  max-width: 1440px;
  margin: 0 auto 36px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.brand-mark {
  font-size: 16px;
  font-weight: 800;
  letter-spacing: 0;
}

.nav-meta {
  display: flex;
  gap: 18px;
  color: #666666;
  font-size: 13px;
}

.nav-meta a {
  color: inherit;
  text-decoration: none;
  transition: color 0.18s ease;
}

.nav-meta a:hover,
.nav-meta a:focus-visible {
  color: #165dff;
}

.nav-meta a:focus-visible {
  outline: 2px solid #165dff;
  outline-offset: 4px;
}

.home-workspace {
  max-width: 1440px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(480px, 0.82fr) minmax(640px, 1.18fr);
  gap: 32px;
  align-items: start;
}

.destination-feature {
  position: relative;
  min-width: 0;
  min-height: 560px;
  overflow: hidden;
  background: #111111;
  color: #ffffff;
}

.destination-feature__image,
.destination-feature__placeholder,
.destination-feature__scrim {
  position: absolute;
  inset: 0;
}

.destination-feature__image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: 36% center;
}

.destination-feature__placeholder {
  display: grid;
  place-items: center;
  background: #dfe3e8;
  color: #59616c;
  font-size: 40px;
  font-weight: 700;
}

.destination-feature__scrim {
  background: rgba(17, 17, 17, 0.58);
}

.destination-feature__content {
  position: relative;
  z-index: 1;
  min-height: 560px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: flex-start;
  padding: clamp(28px, 4vw, 48px);
}

.destination-feature__content h1 {
  max-width: 520px;
  margin: 0;
  font-size: 52px;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: 0;
  overflow-wrap: anywhere;
}

.destination-feature__content > p:not(.eyebrow) {
  max-width: 460px;
  margin: 20px 0 24px;
  color: rgba(255, 255, 255, 0.88);
  font-size: 16px;
  line-height: 1.7;
}

.destination-feature__duration {
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.42);
  color: #ffffff;
  font-size: 13px;
  font-weight: 700;
}

.eyebrow {
  margin: 0 0 12px;
  color: #165dff;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
}

.destination-feature .eyebrow {
  color: rgba(255, 255, 255, 0.82);
}

.request-card {
  min-width: 0;
  height: 100%;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 16px 38px rgba(17, 17, 17, 0.07);
  scroll-margin-top: 24px;
}

.request-card :deep(.ant-card-body) {
  height: 100%;
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
  color: #111111;
}

.card-heading p {
  margin: 0;
  color: #666666;
}

.data-badge {
  flex: 0 0 auto;
  padding: 6px 10px;
  border-radius: 4px;
  color: #165dff;
  background: #ffffff;
  border: 1px solid rgba(22, 93, 255, 0.18);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.form-section {
  padding: 18px 0 6px;
  border-top: 1px solid #e0e0e0;
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
  color: #111111;
  font-size: 15px;
  font-weight: 800;
}

.section-heading small {
  color: #666666;
}

.form-label {
  color: #555555;
  font-weight: 650;
}

.quiet-input :deep(.ant-input),
.quiet-input :deep(.ant-picker),
.quiet-textarea :deep(.ant-input),
.quiet-select :deep(.ant-select-selector) {
  border-color: #e5e5e5 !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

.quiet-input :deep(.ant-input:hover),
.quiet-input :deep(.ant-picker:hover),
.quiet-textarea :deep(.ant-input:hover),
.quiet-select:hover :deep(.ant-select-selector) {
  border-color: #165dff !important;
}

.quiet-input :deep(.ant-input:focus),
.quiet-input :deep(.ant-picker-focused),
.quiet-textarea :deep(.ant-input:focus),
.quiet-select :deep(.ant-select-focused .ant-select-selector) {
  border-color: #165dff !important;
  box-shadow: 0 0 0 3px rgba(22, 93, 255, 0.1) !important;
}

.days-pill {
  height: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0;
  border-radius: 6px;
  color: #165dff;
  background: #f0f7ff;
  border: 1px solid rgba(22, 93, 255, 0.18);
}

.days-pill strong {
  font-size: 18px;
  line-height: 1;
}

.days-pill span {
  max-width: 100%;
  font-size: 10px;
  line-height: 1.2;
  white-space: nowrap;
}

.preference-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.preference-pill {
  position: relative;
  margin: 0;
}

.preference-pill :deep(.ant-checkbox) {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.preference-pill:focus-within {
  border-radius: 4px;
  outline: 3px solid #165dff;
  outline-offset: 3px;
}

.preference-pill :deep(span:last-child) {
  display: inline-flex;
  padding: 7px 12px;
  border: 1px solid #e5e5e5;
  border-radius: 4px;
  color: #555555;
  background: #ffffff;
  transition: all 0.18s ease;
}

.preference-pill :deep(.ant-checkbox-checked + span) {
  color: #165dff;
  background: #f0f7ff;
  border-color: rgba(22, 93, 255, 0.32);
}

.submit-row {
  margin-top: 20px;
}

.plan-summary {
  display: flex;
  flex-wrap: wrap;
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

.submit-button {
  height: 46px;
  border-radius: 6px;
  background: #165dff;
  border-color: #165dff;
  font-weight: 800;
  box-shadow: none;
}

.submit-button:hover,
.submit-button:focus {
  background: #0b46cc !important;
  border-color: #0b46cc !important;
}

.loading-panel {
  padding: 14px;
  background: #fafafa;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

.loading-copy {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
  color: #666666;
}

.loading-copy strong {
  color: #165dff;
}

.inspiration-section,
.workflow-band {
  max-width: 1440px;
  margin: 72px auto 0;
  scroll-margin-top: 24px;
}

.section-intro {
  max-width: 620px;
}

.section-intro h2,
.workflow-band h2 {
  margin: 0;
  color: #111111;
  font-size: 30px;
  line-height: 1.2;
  letter-spacing: 0;
}

.section-intro > p:last-child {
  margin: 12px 0 0;
  color: #666666;
  font-size: 15px;
  line-height: 1.7;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.preset-tile {
  min-width: 0;
  display: flex;
  flex-direction: column;
  padding: 0;
  overflow: hidden;
  border: 1px solid #dedede;
  border-radius: 8px;
  background: #ffffff;
  color: #111111;
  cursor: pointer;
  font: inherit;
  text-align: left;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.preset-tile:hover {
  border-color: #9b9b9b;
}

.preset-tile.is-selected {
  border-color: #165dff;
  box-shadow: 0 0 0 2px rgba(22, 93, 255, 0.24);
}

.preset-tile:focus-visible {
  outline: 3px solid #165dff;
  outline-offset: 3px;
}

.preset-tile__media {
  position: relative;
  display: block;
  width: 100%;
  aspect-ratio: 4 / 3;
  flex: 0 0 auto;
  overflow: hidden;
  background: #e6e9ed;
}

.preset-tile__media img,
.preset-tile__placeholder {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.preset-tile__media img {
  object-fit: cover;
}

.preset-tile__placeholder {
  display: grid;
  place-items: center;
  background: #e2e6eb;
  color: #59616c;
  font-size: 22px;
  font-weight: 700;
}

.preset-tile__scrim {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  height: 42px;
  background: rgba(17, 17, 17, 0.58);
}

.preset-tile__meta {
  position: absolute;
  z-index: 1;
  right: 16px;
  bottom: 12px;
  left: 16px;
  overflow: hidden;
  color: #ffffff;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preset-tile__content {
  min-height: 144px;
  display: flex;
  flex: 1;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 18px;
}

.preset-tile__content strong {
  font-size: 20px;
  line-height: 1.25;
}

.preset-tile__description {
  color: #666666;
  font-size: 13px;
  line-height: 1.55;
}

.preset-tile__command {
  margin-top: auto;
  color: #165dff;
  font-size: 13px;
  font-weight: 700;
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
  min-width: 0;
  display: grid;
  gap: 8px;
  color: #555555;
  font-size: 13px;
  line-height: 1.5;
}

.workflow-steps strong {
  color: #165dff;
  font-size: 13px;
}

@media (max-width: 1230px) {
  .home-workspace {
    grid-template-columns: 1fr;
  }
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
    gap: 16px;
    overflow-x: auto;
    overscroll-behavior-inline: contain;
    scroll-snap-type: x mandatory;
    scroll-padding-inline: 6px;
    padding: 6px 6px 12px;
    scrollbar-width: thin;
  }

  .preset-tile {
    flex: 0 0 300px;
    scroll-snap-align: start;
  }

  .workflow-band {
    align-items: start;
  }

  .workflow-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .home-page {
    padding: 20px 14px 32px;
  }

  .home-nav {
    margin-bottom: 28px;
  }

  .nav-meta {
    flex: 0 0 auto;
    gap: 10px;
  }

  .destination-feature__content {
    padding: 24px;
  }

  .destination-feature__content h1 {
    font-size: 38px;
  }

  .destination-feature__content > p:not(.eyebrow) {
    font-size: 15px;
  }

  .card-heading {
    flex-direction: column;
  }

  .request-card :deep(.ant-card-body) {
    padding: 18px;
  }

  .plan-summary__label {
    flex-basis: 100%;
  }

  .plan-summary__item {
    padding-left: 0;
    border-left: 0;
  }

  .inspiration-section,
  .workflow-band {
    margin-top: 56px;
  }

  .section-intro h2,
  .workflow-band h2 {
    font-size: 26px;
  }

  .preset-tile {
    flex-basis: min(78vw, 300px);
  }

  .workflow-steps {
    grid-template-columns: 1fr;
  }

  .workflow-steps li {
    grid-template-columns: 32px minmax(0, 1fr);
    align-items: baseline;
    gap: 12px;
  }
}

@media (max-width: 360px) {
  .destination-feature__content {
    padding: 16px;
  }

  .destination-feature__content h1 {
    font-size: 30px;
  }

  .destination-feature__content > p:not(.eyebrow) {
    margin: 8px 0 10px;
    font-size: 13px;
    line-height: 1.45;
  }

  .destination-feature .eyebrow {
    margin-bottom: 6px;
    font-size: 11px;
  }

  .destination-feature__duration {
    padding-top: 8px;
    font-size: 12px;
    line-height: 1.3;
  }
}
</style>

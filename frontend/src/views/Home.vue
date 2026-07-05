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
              <a-col :xs="24" :md="8">
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
              <a-col :xs="24" :md="8">
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

<script setup lang="ts">
import { reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { generateTripPlanStream } from '@/services/api'
import type { TripFormData } from '@/types'
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

watch([() => formData.start_date, () => formData.end_date], ([start, end]) => {
  if (start && end) {
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
</style>

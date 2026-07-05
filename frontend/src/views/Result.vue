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
              <a-menu-item v-if="tripPlan.budget" key="budget">预算</a-menu-item>
              <a-menu-item key="map">地图</a-menu-item>
              <a-sub-menu key="days" title="每日行程">
                <a-menu-item v-for="(day, index) in tripPlan.days" :key="`day-${index}`">
                  第{{ day.day_index + 1 }}天
                </a-menu-item>
              </a-sub-menu>
              <a-menu-item v-if="tripPlan.weather_info && tripPlan.weather_info.length > 0" key="weather">
                天气
              </a-menu-item>
            </a-menu>
          </a-affix>
        </aside>

        <section class="main-content">
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
              <a-collapse-panel
                v-for="(day, index) in tripPlan.days"
                :key="index"
                :id="`day-${index}`"
              >
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
                  <article
                    v-for="(item, attrIndex) in day.attractions"
                    :key="`${day.day_index}-${item.name}-${attrIndex}`"
                    class="attraction-row"
                  >
                    <div class="attraction-media">
                      <img
                        :src="getAttractionImage(item.name, attrIndex)"
                        :alt="item.name"
                        @error="handleImageError"
                      />
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
                        <a-space wrap>
                          <a-button
                            size="small"
                            @click="moveAttraction(day.day_index, attrIndex, 'up')"
                            :disabled="attrIndex === 0"
                          >
                            上移
                          </a-button>
                          <a-button
                            size="small"
                            @click="moveAttraction(day.day_index, attrIndex, 'down')"
                            :disabled="attrIndex === day.attractions.length - 1"
                          >
                            下移
                          </a-button>
                          <a-button size="small" danger @click="deleteAttraction(day.day_index, attrIndex)">
                            删除
                          </a-button>
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

          <section
            id="weather"
            v-if="tripPlan.weather_info && tripPlan.weather_info.length > 0"
            class="panel weather-panel"
          >
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

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { DownOutlined } from '@ant-design/icons-vue'
import AMapLoader from '@amap/amap-jsapi-loader'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import { getAttractionPhoto } from '@/services/api'
import type { TripPlan } from '@/types'

const router = useRouter()
const tripPlan = ref<TripPlan | null>(null)
const editMode = ref(false)
const originalPlan = ref<TripPlan | null>(null)
const attractionPhotos = ref<Record<string, string>>({})
const activeSection = ref('overview')
const activeDays = ref<number[]>([0])
let map: any = null

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

onMounted(async () => {
  const data = sessionStorage.getItem('tripPlan')
  if (data) {
    tripPlan.value = JSON.parse(data)
    await loadAttractionPhotos()
    await nextTick()
    initMap()
  }
})

const goBack = () => {
  router.push('/')
}

const scrollToSection = ({ key }: { key: string }) => {
  activeSection.value = key
  const element = document.getElementById(key)
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

const toggleEditMode = () => {
  editMode.value = true
  originalPlan.value = JSON.parse(JSON.stringify(tripPlan.value))
  message.info('进入编辑模式')
}

const saveChanges = () => {
  editMode.value = false
  if (tripPlan.value) {
    sessionStorage.setItem('tripPlan', JSON.stringify(tripPlan.value))
  }
  message.success('修改已保存')

  if (map) {
    map.destroy()
  }
  nextTick(() => {
    initMap()
  })
}

const cancelEdit = () => {
  if (originalPlan.value) {
    tripPlan.value = JSON.parse(JSON.stringify(originalPlan.value))
  }
  editMode.value = false
  message.info('已取消编辑')
}

const deleteAttraction = (dayIndex: number, attrIndex: number) => {
  if (!tripPlan.value) return

  const day = tripPlan.value.days[dayIndex]
  if (day.attractions.length <= 1) {
    message.warning('每天至少需要保留一个景点')
    return
  }

  day.attractions.splice(attrIndex, 1)
  message.success('景点已删除')
}

const moveAttraction = (dayIndex: number, attrIndex: number, direction: 'up' | 'down') => {
  if (!tripPlan.value) return

  const day = tripPlan.value.days[dayIndex]
  const attractions = day.attractions

  if (direction === 'up' && attrIndex > 0) {
    [attractions[attrIndex], attractions[attrIndex - 1]] = [attractions[attrIndex - 1], attractions[attrIndex]]
  } else if (direction === 'down' && attrIndex < attractions.length - 1) {
    [attractions[attrIndex], attractions[attrIndex + 1]] = [attractions[attrIndex + 1], attractions[attrIndex]]
  }
}

const getMealLabel = (type: string): string => {
  const labels: Record<string, string> = {
    breakfast: '早餐',
    lunch: '午餐',
    dinner: '晚餐',
    snack: '小吃'
  }
  return labels[type] || type
}

const loadAttractionPhotos = async () => {
  if (!tripPlan.value) return

  const promises: Promise<void>[] = []

  tripPlan.value.days.forEach(day => {
    day.attractions.forEach(attraction => {
      const promise = getAttractionPhoto(attraction.name)
        .then(photoUrl => {
          if (photoUrl) {
            attractionPhotos.value[attraction.name] = photoUrl
          }
        })
        .catch(err => {
          console.error(`获取${attraction.name}图片失败:`, err)
        })

      promises.push(promise)
    })
  })

  await Promise.all(promises)
}

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

const handleImageError = (event: Event) => {
  const img = event.target as HTMLImageElement
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260"><rect width="400" height="260" fill="#eef2f0"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#66736f">图片暂不可用</text></svg>'
  img.src = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`
}

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

const prepareExportContainer = async (): Promise<HTMLCanvasElement> => {
  const element = document.querySelector('.main-content') as HTMLElement
  if (!element) throw new Error('未找到内容元素')

  const container = document.createElement('div')
  container.style.width = element.offsetWidth + 'px'
  container.style.backgroundColor = '#f6f8f7'
  container.style.padding = '20px'
  container.innerHTML = element.innerHTML

  const mapContainer = document.getElementById('amap-container')
  if (mapContainer && map) {
    const mapCanvas = mapContainer.querySelector('canvas')
    if (mapCanvas) {
      const exportMapContainer = container.querySelector('#amap-container')
      if (exportMapContainer) {
        exportMapContainer.innerHTML = `<img src="${mapCanvas.toDataURL('image/png')}" style="width:100%;height:100%;object-fit:cover" />`
      }
    }
  }

  applyExportStyles(container)

  container.style.position = 'absolute'
  container.style.left = '-9999px'
  document.body.appendChild(container)

  const canvas = await html2canvas(container, {
    backgroundColor: '#f6f8f7', scale: 2, logging: false, useCORS: true, allowTaint: true,
  })

  document.body.removeChild(container)
  return canvas
}

const exportAsImage = async () => {
  try {
    message.loading({ content: '正在生成图片...', key: 'export', duration: 0 })
    const canvas = await prepareExportContainer()
    const link = document.createElement('a')
    link.download = `旅行计划_${tripPlan.value?.city}_${new Date().getTime()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
    message.success({ content: '图片导出成功', key: 'export' })
  } catch (error: any) {
    console.error('导出图片失败:', error)
    message.error({ content: `导出图片失败: ${error.message}`, key: 'export' })
  }
}

const exportAsPDF = async () => {
  try {
    message.loading({ content: '正在生成 PDF...', key: 'export', duration: 0 })
    const canvas = await prepareExportContainer()
    const imgData = canvas.toDataURL('image/png')
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
    const imgWidth = 210
    const imgHeight = (canvas.height * imgWidth) / canvas.width

    let heightLeft = imgHeight
    let position = 0
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
    heightLeft -= 297

    while (heightLeft > 0) {
      position = heightLeft - imgHeight
      pdf.addPage()
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
      heightLeft -= 297
    }

    pdf.save(`旅行计划_${tripPlan.value?.city}_${new Date().getTime()}.pdf`)
    message.success({ content: 'PDF 导出成功', key: 'export' })
  } catch (error: any) {
    console.error('导出PDF失败:', error)
    message.error({ content: `导出PDF失败: ${error.message}`, key: 'export' })
  }
}

const initMap = async () => {
  try {
    const amapKey = import.meta.env.VITE_AMAP_WEB_JS_KEY
    if (!amapKey) {
      message.error('地图加载失败: 未配置高德 JS API Key')
      return
    }

    const AMap = await AMapLoader.load({
      key: amapKey,
      version: '2.0',
      plugins: ['AMap.Marker', 'AMap.Polyline', 'AMap.InfoWindow']
    })

    map = new AMap.Map('amap-container', {
      zoom: 12,
      center: [120.209903, 30.246566],
      viewMode: '3D'
    })

    addAttractionMarkers(AMap)

    message.success('地图加载成功')
  } catch (error) {
    console.error('地图加载失败:', error)
    message.error('地图加载失败')
  }
}

const addAttractionMarkers = (AMap: any) => {
  if (!tripPlan.value) return

  const markers: any[] = []
  const allAttractions: any[] = []

  tripPlan.value.days.forEach((day, dayIndex) => {
    day.attractions.forEach((attraction, attrIndex) => {
      if (attraction.location && attraction.location.longitude && attraction.location.latitude) {
        allAttractions.push({
          ...attraction,
          dayIndex,
          attrIndex
        })
      }
    })
  })

  allAttractions.forEach((attraction, index) => {
    const marker = new AMap.Marker({
      position: [attraction.location.longitude, attraction.location.latitude],
      title: attraction.name,
      label: {
        content: `<div style="background:#0f766e;color:#fff;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:700;box-shadow:0 4px 10px rgba(15,118,110,.24);">${index + 1}</div>`,
        offset: new AMap.Pixel(0, -30)
      }
    })

    const infoWindow = new AMap.InfoWindow({
      content: `
        <div style="padding: 10px;">
          <h4 style="margin: 0 0 8px 0;">${attraction.name}</h4>
          <p style="margin: 4px 0;"><strong>地址:</strong> ${attraction.address}</p>
          <p style="margin: 4px 0;"><strong>游览时长:</strong> ${attraction.visit_duration}分钟</p>
          <p style="margin: 4px 0;"><strong>描述:</strong> ${attraction.description}</p>
          <p style="margin: 4px 0; color: #0f766e;"><strong>第${attraction.dayIndex + 1}天 景点${attraction.attrIndex + 1}</strong></p>
        </div>
      `,
      offset: new AMap.Pixel(0, -30)
    })

    marker.on('click', () => {
      infoWindow.open(map, marker.getPosition())
    })

    markers.push(marker)
  })

  map.add(markers)

  if (allAttractions.length > 0) {
    map.setFitView(markers)
  }

  drawRoutes(AMap, allAttractions)
}

const drawRoutes = (AMap: any, attractions: any[]) => {
  if (attractions.length < 2) return

  const dayGroups: any = {}
  attractions.forEach(attr => {
    if (!dayGroups[attr.dayIndex]) {
      dayGroups[attr.dayIndex] = []
    }
    dayGroups[attr.dayIndex].push(attr)
  })

  Object.values(dayGroups).forEach((dayAttractions: any) => {
    if (dayAttractions.length < 2) return

    const path = dayAttractions.map((attr: any) => [
      attr.location.longitude,
      attr.location.latitude
    ])

    const polyline = new AMap.Polyline({
      path: path,
      strokeColor: '#0f766e',
      strokeWeight: 4,
      strokeOpacity: 0.78,
      strokeStyle: 'solid',
      showDir: true
    })

    map.add(polyline)
  })
}
</script>

<style scoped>
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
</style>

import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '')

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 650000, // 约5.8分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 流式生成旅行计划 (SSE)
 * onProgress: 收到进度事件时回调
 * 返回完整的 TripPlanResponse
 */
export async function generateTripPlanStream(
  formData: TripFormData,
  onProgress: (step: string, message: string) => void
): Promise<TripPlanResponse> {
  const response = await fetch(`${API_BASE_URL}/api/trip/plan-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('响应体不支持流式读取')

  const decoder = new TextDecoder()
  let buffer = ''

  const processFrame = (frame: string): TripPlanResponse | null => {
    let eventType = 'message'
    const dataLines: string[] = []

    for (const rawLine of frame.split(/\r?\n/)) {
      const line = rawLine.trimEnd()
      if (!line || line.startsWith(':')) continue
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart())
      }
    }

    if (dataLines.length === 0) return null

    const data = JSON.parse(dataLines.join('\n'))
    if (eventType === 'progress') {
      onProgress(data.step, data.message)
      return null
    }
    if (eventType === 'result') {
      return data as TripPlanResponse
    }
    if (eventType === 'error') {
      throw new Error(data.message || '生成失败')
    }

    return null
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split(/\r?\n\r?\n/)
    buffer = frames.pop() || ''

    for (const frame of frames) {
      const result = processFrame(frame)
      if (result) return result
    }
  }

  buffer += decoder.decode()
  if (buffer.trim()) {
    const result = processFrame(buffer)
    if (result) return result
  }

  throw new Error('流式响应未返回结果')
}

/**
 * 生成旅行计划 (非流式，保留兼容)
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export async function getAttractionPhoto(name: string): Promise<string | null> {
  try {
    const response = await apiClient.get('/api/poi/photo', {
      params: { name }
    })
    return response.data?.data?.photo_url || null
  } catch (error: any) {
    console.error('获取景点图片失败:', error)
    return null
  }
}

export default apiClient

import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '')

const DEFAULT_ABSOLUTE_TIMEOUT_MS = 310_000
const DEFAULT_INACTIVITY_TIMEOUT_MS = 45_000

export interface TripStreamOptions {
  signal?: AbortSignal
  absoluteTimeoutMs?: number
  inactivityTimeoutMs?: number
}

export class TripStreamError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly retryAfter?: number,
  ) {
    super(message)
    this.name = 'TripStreamError'
  }
}

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
  onProgress: (step: string, message: string) => void,
  options: TripStreamOptions = {},
): Promise<TripPlanResponse> {
  const controller = new AbortController()
  const absoluteTimeoutMs = options.absoluteTimeoutMs ?? DEFAULT_ABSOLUTE_TIMEOUT_MS
  const inactivityTimeoutMs = options.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS
  let abortKind: 'external' | 'timeout' | null = null
  let absoluteTimer: ReturnType<typeof setTimeout> | undefined
  let inactivityTimer: ReturnType<typeof setTimeout> | undefined
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined
  const decoder = new TextDecoder()
  let buffer = ''

  const abort = (kind: 'external' | 'timeout') => {
    if (controller.signal.aborted) return
    abortKind = kind
    controller.abort()
  }

  const externalAbort = () => abort('external')
  if (options.signal?.aborted) externalAbort()
  else options.signal?.addEventListener('abort', externalAbort, { once: true })

  const resetInactivityTimer = () => {
    if (inactivityTimer) clearTimeout(inactivityTimer)
    inactivityTimer = setTimeout(() => abort('timeout'), inactivityTimeoutMs)
  }

  absoluteTimer = setTimeout(() => abort('timeout'), absoluteTimeoutMs)
  resetInactivityTimer()

  let rejectAbort: ((error: TripStreamError) => void) | undefined
  const abortPromise = new Promise<never>((_resolve, reject) => {
    rejectAbort = reject
  })
  const onInternalAbort = () => {
    rejectAbort?.(
      abortKind === 'external'
        ? new TripStreamError('TRIP_CANCELLED', 'Trip request was cancelled.')
        : new TripStreamError('TRIP_TIMEOUT', 'Trip planning timed out. Please try again.'),
    )
  }
  controller.signal.addEventListener('abort', onInternalAbort, { once: true })
  if (controller.signal.aborted) onInternalAbort()

  const withAbort = <T>(promise: Promise<T>): Promise<T> => Promise.race([promise, abortPromise])

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
      throw new TripStreamError(data.code || 'TRIP_FAILED', data.message || 'Trip planning failed.')
    }

    return null
  }

  try {
    const response = await withAbort(fetch(`${API_BASE_URL}/api/trip/plan-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
      signal: controller.signal,
    }))

    if (!response.ok) {
      let detail: { code?: string; message?: string } = {}
      try {
        const payload = await withAbort(response.json())
        detail = payload?.detail || {}
      } catch (error) {
        if (error instanceof TripStreamError) throw error
        // Fall back to the HTTP status when the response body is not JSON.
      }
      const retryAfterHeader = response.headers.get('Retry-After')
      const parsedRetryAfter = retryAfterHeader ? Number.parseInt(retryAfterHeader, 10) : Number.NaN
      throw new TripStreamError(
        detail.code || `HTTP_${response.status}`,
        detail.message || `HTTP ${response.status}: ${response.statusText}`,
        Number.isFinite(parsedRetryAfter) ? parsedRetryAfter : undefined,
      )
    }

    reader = response.body?.getReader()
    if (!reader) throw new TripStreamError('TRIP_FAILED', 'Streaming response is unavailable.')

    while (true) {
      const { done, value } = await withAbort(reader.read())
      if (done) break

      resetInactivityTimer()
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

    throw new TripStreamError('TRIP_FAILED', 'Streaming response ended without a result.')
  } catch (error) {
    if (error instanceof TripStreamError) throw error
    if (controller.signal.aborted || (error instanceof DOMException && error.name === 'AbortError')) {
      throw abortKind === 'external'
        ? new TripStreamError('TRIP_CANCELLED', 'Trip request was cancelled.')
        : new TripStreamError('TRIP_TIMEOUT', 'Trip planning timed out. Please try again.')
    }
    throw new TripStreamError(
      'TRIP_FAILED',
      error instanceof Error ? error.message : 'Trip planning failed.',
    )
  } finally {
    if (absoluteTimer) clearTimeout(absoluteTimer)
    if (inactivityTimer) clearTimeout(inactivityTimer)
    options.signal?.removeEventListener('abort', externalAbort)
    controller.signal.removeEventListener('abort', onInternalAbort)
    if (reader) {
      try {
        await reader.cancel()
      } catch {
        // Cleanup failures must not replace the request outcome.
      }
      try {
        reader.releaseLock()
      } catch {
        // Cleanup failures must not replace the request outcome.
      }
    }
  }
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

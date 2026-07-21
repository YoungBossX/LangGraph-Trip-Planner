import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TripFormData } from '@/types'
import { generateTripPlanStream, TripStreamError } from './api'

const formData: TripFormData = {
  city: 'Hangzhou',
  start_date: '2026-03-01',
  end_date: '2026-03-01',
  travel_days: 1,
  transportation: 'transit',
  accommodation: 'hotel',
  preferences: ['history'],
  free_text_input: '',
}

type ReadResult = ReadableStreamReadResult<Uint8Array>

class ControlledReader {
  private pending: Array<(result: ReadResult) => void> = []
  private queued: ReadResult[] = []
  private readonly encoder = new TextEncoder()

  read = vi.fn(() => {
    const queued = this.queued.shift()
    if (queued) return Promise.resolve(queued)
    return new Promise<ReadResult>((resolve) => this.pending.push(resolve))
  })

  cancel = vi.fn(async () => {
    this.resolve({ done: true, value: undefined })
  })

  releaseLock = vi.fn()

  push(text: string) {
    this.resolve({ done: false, value: this.encoder.encode(text) })
  }

  close() {
    this.resolve({ done: true, value: undefined })
  }

  private resolve(result: ReadResult) {
    const resolve = this.pending.shift()
    if (resolve) resolve(result)
    else this.queued.push(result)
  }
}

function streamResponse(reader: ControlledReader): Response {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers(),
    body: { getReader: () => reader },
  } as unknown as Response
}

function installStreamFetch(reader: ControlledReader) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse(reader)))
}

async function flushAsync() {
  await Promise.resolve()
  await Promise.resolve()
}

describe('generateTripPlanStream', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('preserves progress and result framing and cleans the reader', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const progress = vi.fn()
    const promise = generateTripPlanStream(formData, progress)
    await flushAsync()

    reader.push('event: progress\ndata: {"step":"search_attractions","message":"Searching"}\n\n')
    await flushAsync()
    reader.push('event: result\ndata: {"success":true,"message":"ok","data":{"city":"Hangzhou"}}\n\n')

    await expect(promise).resolves.toMatchObject({ success: true, data: { city: 'Hangzhou' } })
    expect(progress).toHaveBeenCalledWith('search_attractions', 'Searching')
    expect(reader.cancel).toHaveBeenCalledOnce()
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('maps external abort to TRIP_CANCELLED and cleans the reader', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const controller = new AbortController()
    const promise = generateTripPlanStream(formData, vi.fn(), { signal: controller.signal })
    await flushAsync()

    controller.abort()

    await expect(promise).rejects.toMatchObject({ code: 'TRIP_CANCELLED' })
    expect(reader.cancel).toHaveBeenCalledOnce()
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('enforces the absolute timeout and cleans the reader', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn(), {
      absoluteTimeoutMs: 310_000,
      inactivityTimeoutMs: 400_000,
    })
    await flushAsync()
    const rejection = expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT' })

    await vi.advanceTimersByTimeAsync(310_000)

    await rejection
    expect(reader.cancel).toHaveBeenCalledOnce()
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('uses the default absolute timeout', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn(), { inactivityTimeoutMs: 400_000 })
    await flushAsync()
    const rejection = expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT' })

    await vi.advanceTimersByTimeAsync(310_000)

    await rejection
  })

  it('resets inactivity timeout for heartbeat-only chunks', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn(), {
      absoluteTimeoutMs: 200_000,
      inactivityTimeoutMs: 45_000,
    })
    await flushAsync()

    await vi.advanceTimersByTimeAsync(40_000)
    reader.push(': heartbeat\n\n')
    await flushAsync()
    await vi.advanceTimersByTimeAsync(40_000)
    reader.push('event: result\ndata: {"success":true,"message":"ok","data":{"city":"Hangzhou"}}\n\n')

    await expect(promise).resolves.toMatchObject({ success: true })
  })

  it('maps inactivity timeout to TRIP_TIMEOUT', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn(), {
      absoluteTimeoutMs: 200_000,
      inactivityTimeoutMs: 45_000,
    })
    await flushAsync()
    const rejection = expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT' })

    await vi.advanceTimersByTimeAsync(45_000)

    await rejection
    expect(reader.cancel).toHaveBeenCalledOnce()
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('uses the default inactivity timeout', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn(), { absoluteTimeoutMs: 200_000 })
    await flushAsync()
    const rejection = expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT' })

    await vi.advanceTimersByTimeAsync(45_000)

    await rejection
  })

  it('parses stable HTTP errors and Retry-After', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      headers: new Headers({ 'Retry-After': '17' }),
      json: vi.fn().mockResolvedValue({
        detail: { code: 'RATE_LIMITED', message: 'Slow down.' },
      }),
    } as unknown as Response))

    await expect(generateTripPlanStream(formData, vi.fn())).rejects.toMatchObject({
      code: 'RATE_LIMITED',
      message: 'Slow down.',
      retryAfter: 17,
    })
  })

  it('applies timeout classification while reading an HTTP error body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      headers: new Headers(),
      json: () => new Promise<never>(() => undefined),
    } as unknown as Response))
    const promise = generateTripPlanStream(formData, vi.fn(), {
      absoluteTimeoutMs: 100,
      inactivityTimeoutMs: 1_000,
    })
    const rejection = expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT' })

    await vi.advanceTimersByTimeAsync(100)

    await rejection
  })

  it('preserves server SSE error code and message and cleans the reader', async () => {
    const reader = new ControlledReader()
    installStreamFetch(reader)
    const promise = generateTripPlanStream(formData, vi.fn())
    await flushAsync()

    reader.push('event: error\ndata: {"code":"TRIP_TIMEOUT","message":"Server deadline"}\n\n')

    await expect(promise).rejects.toMatchObject({ code: 'TRIP_TIMEOUT', message: 'Server deadline' })
    expect(reader.cancel).toHaveBeenCalledOnce()
    expect(reader.releaseLock).toHaveBeenCalledOnce()
  })

  it('exports typed stream errors', () => {
    const error = new TripStreamError('TRIP_FAILED', 'Failed', 3)

    expect(error).toBeInstanceOf(Error)
    expect(error.code).toBe('TRIP_FAILED')
    expect(error.retryAfter).toBe(3)
  })
})

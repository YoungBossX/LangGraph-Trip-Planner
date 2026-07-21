import { describe, expect, it } from 'vitest'

import { createTripRequestLifecycle } from './tripRequestLifecycle'

describe('createTripRequestLifecycle', () => {
  it('aborts the previous request when a new request begins', () => {
    const lifecycle = createTripRequestLifecycle()
    const first = lifecycle.begin()
    const second = lifecycle.begin()

    expect(first.signal.aborted).toBe(true)
    expect(second.signal.aborted).toBe(false)
  })

  it('does not let a stale finish clear the active request', () => {
    const lifecycle = createTripRequestLifecycle()
    const first = lifecycle.begin()
    const second = lifecycle.begin()

    expect(lifecycle.finish(first)).toBe(false)
    lifecycle.cancel()

    expect(second.signal.aborted).toBe(true)
  })

  it('finishes only the matching active request', () => {
    const lifecycle = createTripRequestLifecycle()
    const controller = lifecycle.begin()

    expect(lifecycle.finish(controller)).toBe(true)
    lifecycle.cancel()

    expect(controller.signal.aborted).toBe(false)
  })

  it('cancels idempotently', () => {
    const lifecycle = createTripRequestLifecycle()
    const controller = lifecycle.begin()

    lifecycle.cancel()
    lifecycle.cancel()

    expect(controller.signal.aborted).toBe(true)
  })
})

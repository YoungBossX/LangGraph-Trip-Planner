import { describe, expect, it } from 'vitest'
import { reconcileSelectedPresetId } from './presetSelection'

const presets = [
  { id: 'hangzhou', city: '杭州' },
  { id: 'beijing', city: '北京' },
] as const

describe('reconcileSelectedPresetId', () => {
  it('keeps the selected preset when the city still matches it', () => {
    expect(reconcileSelectedPresetId('hangzhou', '杭州', presets)).toBe('hangzhou')
  })

  it('clears the selected preset when the city diverges from it', () => {
    expect(reconcileSelectedPresetId('hangzhou', '北京', presets)).toBeNull()
  })
})

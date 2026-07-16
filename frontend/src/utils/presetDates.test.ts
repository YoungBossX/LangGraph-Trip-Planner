import dayjs from 'dayjs'
import { describe, expect, it } from 'vitest'
import { tripPresets } from '../data/tripPresets'
import { suggestPresetEndDate } from './presetDates'

describe('tripPresets', () => {
  it('keeps the approved city display order', () => {
    expect(tripPresets.map((preset) => preset.city)).toEqual(['杭州', '北京', '上海', '长沙'])
  })

  it('keeps the approved preset identities and recommended durations', () => {
    expect(tripPresets.map(({ id, city, recommendedDays }) => ({ id, city, recommendedDays }))).toEqual([
      { id: 'hangzhou', city: '杭州', recommendedDays: 3 },
      { id: 'beijing', city: '北京', recommendedDays: 4 },
      { id: 'shanghai', city: '上海', recommendedDays: 3 },
      { id: 'changsha', city: '长沙', recommendedDays: 3 },
    ])
  })

  it('keeps every preset id unique', () => {
    const ids = tripPresets.map((preset) => preset.id)

    expect(new Set(ids).size).toBe(tripPresets.length)
  })

  it('keeps every recommended duration within the allowed range', () => {
    expect(tripPresets.every((preset) => preset.recommendedDays >= 1 && preset.recommendedDays <= 30)).toBe(true)
  })

  it('keeps the source-cleared local responsive image slots enabled', () => {
    expect(tripPresets.map(({ imageSrc, imageSrcSet, imageAvailable, imageAlt }) => ({ imageSrc, imageSrcSet, imageAvailable, imageAlt }))).toEqual([
      {
        imageSrc: '/inspiration/hangzhou.webp',
        imageSrcSet: '/inspiration/hangzhou-800.webp 800w, /inspiration/hangzhou.webp 1600w',
        imageAvailable: true,
        imageAlt: '西湖岸边的杭州城市风景',
      },
      {
        imageSrc: '/inspiration/beijing.webp',
        imageSrcSet: '/inspiration/beijing-800.webp 800w, /inspiration/beijing.webp 1600w',
        imageAvailable: true,
        imageAlt: '天坛祈年殿古建筑',
      },
      {
        imageSrc: '/inspiration/shanghai.webp',
        imageSrcSet: '/inspiration/shanghai-800.webp 800w, /inspiration/shanghai.webp 1600w',
        imageAvailable: true,
        imageAlt: '上海外滩与陆家嘴城市天际线',
      },
      {
        imageSrc: '/inspiration/changsha.webp',
        imageSrcSet: '/inspiration/changsha-800.webp 800w, /inspiration/changsha.webp 1600w',
        imageAvailable: true,
        imageAlt: '湘江与橘子洲的长沙风景',
      },
    ])
  })
})

describe('suggestPresetEndDate', () => {
  it('suggests an inclusive end date for a valid preset duration', () => {
    const suggestedEndDate = suggestPresetEndDate(dayjs('2026-08-01'), null, 3)

    expect(suggestedEndDate?.format('YYYY-MM-DD')).toBe('2026-08-03')
  })

  it('keeps the same calendar day for a one-day preset duration', () => {
    const suggestedEndDate = suggestPresetEndDate(dayjs('2026-08-01'), null, 1)

    expect(suggestedEndDate?.format('YYYY-MM-DD')).toBe('2026-08-01')
  })

  it('suggests the thirtieth calendar day for a 30-day preset duration', () => {
    const suggestedEndDate = suggestPresetEndDate(dayjs('2026-08-01'), null, 30)

    expect(suggestedEndDate?.format('YYYY-MM-DD')).toBe('2026-08-30')
  })

  it('suggests an inclusive end date for a four-day preset duration', () => {
    const suggestedEndDate = suggestPresetEndDate(dayjs('2026-08-01'), null, 4)

    expect(suggestedEndDate?.format('YYYY-MM-DD')).toBe('2026-08-04')
  })

  it('returns null for an invalid start date', () => {
    expect(suggestPresetEndDate(dayjs('invalid'), null, 3)).toBeNull()
  })

  it('returns null for a fractional preset duration', () => {
    expect(suggestPresetEndDate(dayjs('2026-08-01'), null, 3.5)).toBeNull()
  })

  it('leaves an existing end date untouched', () => {
    const existingEndDate = dayjs('2026-08-05')

    expect(suggestPresetEndDate(dayjs('2026-08-01'), existingEndDate, 3)).toBeNull()
    expect(existingEndDate.format('YYYY-MM-DD')).toBe('2026-08-05')
  })

  it.each([
    ['no start date', null, 3],
    ['a zero-day duration', dayjs('2026-08-01'), 0],
    ['a duration over 30 days', dayjs('2026-08-01'), 31],
  ])('returns null for %s', (_reason, startDate, recommendedDays) => {
    expect(suggestPresetEndDate(startDate, null, recommendedDays)).toBeNull()
  })
})

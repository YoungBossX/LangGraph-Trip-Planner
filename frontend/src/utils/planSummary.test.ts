import { describe, expect, it } from 'vitest'
import { getPlanSummaryItems } from './planSummary'

describe('getPlanSummaryItems', () => {
  it('uses the recommended duration when travel days are not set', () => {
    expect(
      getPlanSummaryItems({
        city: '杭州',
        travelDays: 0,
        recommendedDays: 3,
        transportation: '公共交通',
        accommodation: '舒适型酒店',
      }),
    ).toEqual(['杭州', '3 天建议', '公共交通', '舒适型酒店'])
  })

  it('uses the travel duration when travel days are set', () => {
    expect(
      getPlanSummaryItems({
        city: '北京',
        travelDays: 4,
        recommendedDays: 3,
        transportation: '公共交通',
        accommodation: '舒适型酒店',
      }),
    ).toEqual(['北京', '4 天行程', '公共交通', '舒适型酒店'])
  })
})

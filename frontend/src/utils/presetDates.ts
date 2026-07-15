import type { Dayjs } from 'dayjs'

export function suggestPresetEndDate(
  startDate: Dayjs | null,
  endDate: Dayjs | null,
  recommendedDays: number,
): Dayjs | null {
  if (
    !startDate ||
    !startDate.isValid() ||
    endDate !== null ||
    !Number.isInteger(recommendedDays) ||
    recommendedDays < 1 ||
    recommendedDays > 30
  ) {
    return null
  }

  return startDate.add(recommendedDays - 1, 'day')
}

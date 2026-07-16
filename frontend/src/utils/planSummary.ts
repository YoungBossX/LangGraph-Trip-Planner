export interface PlanSummaryInput {
  city: string
  travelDays: number
  recommendedDays: number
  transportation: string
  accommodation: string
}

export const getPlanSummaryItems = ({
  city,
  travelDays,
  recommendedDays,
  transportation,
  accommodation,
}: PlanSummaryInput): readonly string[] => [
  city,
  travelDays > 0 ? `${travelDays} 天行程` : `${recommendedDays} 天建议`,
  transportation,
  accommodation,
]

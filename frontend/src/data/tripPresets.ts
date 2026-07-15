import type { TripFormData } from '../types'

export type TripPresetId = 'hangzhou' | 'beijing' | 'shanghai' | 'changsha'

export interface TripPreset {
  readonly id: TripPresetId
  readonly city: TripFormData['city']
  readonly title: string
  readonly description: string
  readonly recommendedDays: TripFormData['travel_days']
  readonly transportation: TripFormData['transportation']
  readonly accommodation: TripFormData['accommodation']
  readonly preferences: Readonly<TripFormData['preferences']>
  readonly imageSrc: string
  readonly imageAvailable: boolean
  readonly imageAlt: string
}

export const tripPresets: readonly TripPreset[] = [
  {
    id: 'hangzhou',
    city: '杭州',
    title: '湖畔慢游',
    description: '沿着西湖与老街，留一点时间给茶香。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['历史文化', '自然风光', '休闲'],
    imageSrc: '/inspiration/hangzhou.webp',
    imageAvailable: false,
    imageAlt: '西湖岸边的杭州城市风景',
  },
  {
    id: 'beijing',
    city: '北京',
    title: '古都与新展',
    description: '在古迹与当代展览之间，从容走读北京。',
    recommendedDays: 4,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['历史文化', '艺术'],
    imageSrc: '/inspiration/beijing.webp',
    imageAvailable: false,
    imageAlt: '北京古建筑与城市天际线',
  },
  {
    id: 'shanghai',
    city: '上海',
    title: '城市漫游',
    description: '把街区、展馆与风味小店串成一段漫游。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '舒适型酒店',
    preferences: ['艺术', '购物', '美食'],
    imageSrc: '/inspiration/shanghai.webp',
    imageAvailable: false,
    imageAlt: '上海外滩与陆家嘴城市天际线',
  },
  {
    id: 'changsha',
    city: '长沙',
    title: '晚风与夜宵',
    description: '白天逛城，夜里在烟火气里吃一顿夜宵。',
    recommendedDays: 3,
    transportation: '公共交通',
    accommodation: '经济型酒店',
    preferences: ['美食', '休闲'],
    imageSrc: '/inspiration/changsha.webp',
    imageAvailable: false,
    imageAlt: '长沙夜景与热闹街道',
  },
]

type CityPreset = {
  id: string
  city: string
}

export function reconcileSelectedPresetId<TPreset extends CityPreset>(
  selectedPresetId: TPreset['id'] | null,
  city: string,
  presets: readonly TPreset[],
): TPreset['id'] | null {
  return presets.some((preset) => preset.id === selectedPresetId && preset.city === city)
    ? selectedPresetId
    : null
}

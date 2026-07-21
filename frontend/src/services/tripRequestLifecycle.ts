export interface TripRequestLifecycle {
  begin(): AbortController
  finish(controller: AbortController): boolean
  cancel(): void
}

export function createTripRequestLifecycle(): TripRequestLifecycle {
  let activeController: AbortController | null = null

  return {
    begin() {
      activeController?.abort()
      const controller = new AbortController()
      activeController = controller
      return controller
    },
    finish(controller) {
      if (activeController !== controller) return false
      activeController = null
      return true
    },
    cancel() {
      activeController?.abort()
      activeController = null
    },
  }
}

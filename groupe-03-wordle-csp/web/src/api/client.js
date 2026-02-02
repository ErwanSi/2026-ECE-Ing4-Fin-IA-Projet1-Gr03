const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function fetchAPI(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Erreur API' }))
    throw new Error(error.detail || `Erreur HTTP ${response.status}`)
  }

  return response.json()
}

export async function checkHealth() {
  return fetchAPI('/health')
}

export async function createGame(strategy = 'mixed', solver = 'filtering') {
  return fetchAPI('/game/new', {
    method: 'POST',
    body: JSON.stringify({ strategy, solver }),
  })
}

export async function addConstraint(gameId, guess, feedback) {
  return fetchAPI(`/game/${gameId}/constraint`, {
    method: 'POST',
    body: JSON.stringify({ guess, feedback }),
  })
}

export async function getSuggestion(gameId, limit = 1) {
  return fetchAPI(`/game/${gameId}/suggest?limit=${limit}`)
}

export async function getGameState(gameId) {
  return fetchAPI(`/game/${gameId}/state`)
}

export async function simulateGame(secret, maxTurns = 6) {
  return fetchAPI('/simulate', {
    method: 'POST',
    body: JSON.stringify({ secret, max_turns: maxTurns }),
  })
}

import { useState, useEffect } from 'react'
import './App.css'
import { checkHealth } from './api/client'
import WordleGrid from './components/WordleGrid'
import Controls from './components/Controls'
import Stats from './components/Stats'
import SimulationPanel from './components/SimulationPanel'

function App() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  // État du jeu
  const [gameId, setGameId] = useState(null)
  const [strategy, setStrategy] = useState('mixed')
  const [solver, setSolver] = useState('filtering')
  const [attempts, setAttempts] = useState([])
  const [stats, setStats] = useState({
    candidatesCount: 0,
    totalWords: 0,
    solved: false,
    solution: null
  })

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const healthData = await checkHealth()
        setHealth(healthData)
        setError(null)
      } catch (err) {
        setError('Impossible de se connecter au backend. Assurez-vous que le serveur API est démarré sur http://localhost:8000')
        setHealth(null)
      } finally {
        setLoading(false)
      }
    }

    fetchHealth()
    const interval = setInterval(fetchHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleNewGame = (newGameId, newStats) => {
    setGameId(newGameId)
    setAttempts([])
    setStats({
      ...newStats,
      solverType: solver
    })
  }

  const handleAttemptAdded = (attempt, newStats) => {
    setAttempts(prev => [...prev, attempt])
    setStats(newStats)
  }

  const handleStrategyChange = (newStrategy) => {
    setStrategy(newStrategy)
  }

  const handleSolverChange = (newSolver) => {
    setSolver(newSolver)
  }

  return (
    <div className="app">
      <div className={`health-status ${health ? 'online' : 'offline'}`}>
        {health ? `✓ API Online (${health.word_count} mots)` : '✗ API Offline'}
      </div>

      <header className="app-header">
        <h1>🎯 Wordle CSP Solver Demo</h1>
        <p>Solveur Wordle intelligent basé sur CSP (Constraint Satisfaction Problem)</p>
      </header>

      {loading && <div className="loading">Chargement...</div>}
      
      {error && <div className="error">{error}</div>}
      
      {health && !loading && (
        <div className="main-content">
          <div className="game-section">
            <h2>Mode Interactif</h2>
            <WordleGrid attempts={attempts} />
            <Controls
              gameId={gameId}
              strategy={strategy}
              solver={solver}
              onNewGame={handleNewGame}
              onAttemptAdded={handleAttemptAdded}
              onStrategyChange={handleStrategyChange}
              onSolverChange={handleSolverChange}
              stats={stats}
            />
            <Stats stats={stats} attemptsCount={attempts.length} />
          </div>
          
          <div className="solver-section">
            <h2>Mode Automatique</h2>
            <SimulationPanel />
          </div>
        </div>
      )}
    </div>
  )
}

export default App

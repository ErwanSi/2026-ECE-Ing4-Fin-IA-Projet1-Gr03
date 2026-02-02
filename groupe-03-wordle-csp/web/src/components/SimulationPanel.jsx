import { useState } from 'react'
import { simulateGame } from '../api/client'
import '../styles/SimulationPanel.css'

function SimulationPanel() {
  const [secret, setSecret] = useState('')
  const [solverType, setSolverType] = useState('filtering')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSimulate = async () => {
    if (secret.length !== 5) {
      setError('Le mot secret doit contenir exactement 5 lettres')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await simulateGame(secret.toUpperCase(), 6)
      setResult(response)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="simulation-panel">
      <div className="simulation-input">
        <label htmlFor="solver-sim">Solveur:</label>
        <select
          id="solver-sim"
          value={solverType}
          onChange={(e) => setSolverType(e.target.value)}
          disabled={loading}
        >
          <option value="filtering">CSP Filtrage</option>
          <option value="cpsat">OR-Tools CP-SAT</option>
        </select>

        <label htmlFor="secret">Mot secret (5 lettres):</label>
        <input
          id="secret"
          type="text"
          placeholder="MARDI"
          value={secret}
          onChange={(e) => setSecret(e.target.value.toUpperCase())}
          maxLength={5}
          disabled={loading}
        />
        <button
          className="btn-simulate"
          onClick={handleSimulate}
          disabled={loading || secret.length !== 5}
        >
          {loading ? '⏳ Simulation...' : '🤖 Résoudre automatiquement'}
        </button>
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {result && (
        <div className="simulation-result">
          <h3>Résultat de la simulation</h3>
          
          <div className="result-summary">
            <div className={`result-status ${result.solved ? 'success' : 'failure'}`}>
              {result.solved ? '✅ Résolu !' : '❌ Non résolu'}
            </div>
            <div className="result-info">
              <span>Mot secret: <strong>{result.secret}</strong></span>
              <span>Tentatives: <strong>{result.total_turns}</strong></span>
            </div>
          </div>

          <div className="simulation-turns">
            <h4>Historique des coups:</h4>
            {result.turns.map((turn, index) => (
              <div key={index} className="turn-row">
                <span className="turn-number">#{turn.turn_number}</span>
                <div className="turn-guess">
                  {turn.guess.split('').map((letter, i) => {
                    const feedbackClass = 
                      turn.feedback[i] === 'G' ? 'green' : 
                      turn.feedback[i] === 'Y' ? 'yellow' : 
                      'gray'
                    return (
                      <div key={i} className={`mini-cell ${feedbackClass}`}>
                        {letter}
                      </div>
                    )
                  })}
                </div>
                <span className="turn-candidates">{turn.candidates_remaining} candidats</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default SimulationPanel

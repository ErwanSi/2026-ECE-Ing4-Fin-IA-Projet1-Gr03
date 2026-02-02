import { useState } from 'react'
import { createGame, addConstraint, getSuggestion } from '../api/client'
import '../styles/Controls.css'

function Controls({ gameId, strategy, solver, onNewGame, onAttemptAdded, onStrategyChange, onSolverChange, stats }) {
  const [localStrategy, setLocalStrategy] = useState(strategy)
  const [loading, setLoading] = useState(false)
  const [suggestion, setSuggestion] = useState(null)
  const [manualGuess, setManualGuess] = useState('')
  const [manualFeedback, setManualFeedback] = useState('')
  const [error, setError] = useState(null)

  const handleNewGame = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await createGame(localStrategy, solver)
      onNewGame(response.game_id, {
        candidatesCount: response.candidates_count,
        totalWords: response.total_words,
        solved: false,
        solution: null
      })
      setSuggestion(response.first_suggestion)
      setManualGuess('')
      setManualFeedback('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGetSuggestion = async () => {
    if (!gameId) {
      setError('Créez d\'abord une nouvelle partie')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await getSuggestion(gameId, 1)
      if (response.suggestions && response.suggestions.length > 0) {
        setSuggestion(response.suggestions[0])
      } else {
        setSuggestion(null)
        setError('Aucune suggestion disponible')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAddConstraint = async () => {
    if (!gameId) {
      setError('Créez d\'abord une nouvelle partie')
      return
    }

    if (manualGuess.length !== 5 || manualFeedback.length !== 5) {
      setError('Le guess et le feedback doivent contenir exactement 5 caractères')
      return
    }

    if (!/^[GYB]{5}$/i.test(manualFeedback)) {
      setError('Le feedback doit contenir uniquement G (vert), Y (jaune) ou B (gris)')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const response = await addConstraint(gameId, manualGuess, manualFeedback)
      
      onAttemptAdded(
        {
          guess: manualGuess.toUpperCase(),
          feedback: manualFeedback.toUpperCase(),
          remainingCandidates: response.candidates_remaining
        },
        {
          candidatesCount: response.candidates_remaining,
          totalWords: stats.totalWords,
          solved: response.solved,
          solution: response.solution
        }
      )

      // Réinitialiser les champs
      setManualGuess('')
      setManualFeedback('')
      
      // Obtenir nouvelle suggestion si pas résolu
      if (!response.solved && response.candidates_remaining > 0) {
        handleGetSuggestion()
      } else {
        setSuggestion(null)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleStrategyChange = (e) => {
    const newStrategy = e.target.value
    setLocalStrategy(newStrategy)
    onStrategyChange(newStrategy)
  }

  return (
    <div className="controls">
      {error && <div className="error-message">{error}</div>}
      
      {stats.solved && (
        <div className="success-message">
          ✅ Résolu ! Le mot était : {stats.solution}
        </div>
      )}
      
      <div className="control-group">
        <label htmlFor="solver">Solveur:</label>
        <select
          id="solver"
          value={solver}
          onChange={(e) => onSolverChange(e.target.value)}
          disabled={loading || gameId !== null}
        >
          <option value="filtering">CSP Filtrage (Rapide)</option>
          <option value="cpsat">OR-Tools CP-SAT (Avancé)</option>
        </select>
        <p className="help-text-small">
          Le solveur peut être changé uniquement avant de créer une partie
        </p>
      </div>

      <div className="control-group">
        <label htmlFor="strategy">Stratégie:</label>
        <select
          id="strategy"
          value={localStrategy}
          onChange={handleStrategyChange}
          disabled={loading}
        >
          <option value="mixed">Mixed (Recommandé)</option>
          <option value="frequency">Fréquence</option>
          <option value="entropy">Entropie</option>
          <option value="naive">Naive</option>
        </select>
      </div>

      <div className="control-group">
        <button 
          className="btn-primary" 
          onClick={handleNewGame}
          disabled={loading}
        >
          {loading ? '⏳ Chargement...' : '🎮 Nouvelle Partie'}
        </button>
      </div>

      {gameId && !stats.solved && (
        <>
          <div className="suggestion-box">
            <h3>💡 Suggestion du solveur</h3>
            {suggestion ? (
              <div className="suggestion-word">{suggestion}</div>
            ) : (
              <div className="no-suggestion">Aucune suggestion</div>
            )}
            <button 
              className="btn-secondary" 
              onClick={handleGetSuggestion}
              disabled={loading || stats.candidatesCount === 0}
            >
              Actualiser la suggestion
            </button>
          </div>

          <div className="manual-input">
            <h3>✏️ Ajouter une tentative manuellement</h3>
            <div className="input-row">
              <input
                type="text"
                placeholder="GUESS (5 lettres)"
                value={manualGuess}
                onChange={(e) => setManualGuess(e.target.value.toUpperCase())}
                maxLength={5}
                disabled={loading}
              />
              <input
                type="text"
                placeholder="Feedback (GGYBB)"
                value={manualFeedback}
                onChange={(e) => setManualFeedback(e.target.value.toUpperCase())}
                maxLength={5}
                disabled={loading}
              />
            </div>
            <button 
              className="btn-secondary" 
              onClick={handleAddConstraint}
              disabled={loading || manualGuess.length !== 5 || manualFeedback.length !== 5}
            >
              ➕ Ajouter contrainte
            </button>
            <p className="help-text">
              Feedback: G=Vert (bonne position), Y=Jaune (mauvaise position), B=Gris (absent)
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default Controls

import '../styles/WordleGrid.css'

function WordleGrid({ attempts }) {
  const maxRows = 6

  return (
    <div className="wordle-grid">
      <div className="grid-container">
        {attempts.length === 0 && (
          <p className="placeholder-text">
            Créez une nouvelle partie pour commencer
          </p>
        )}
        
        {attempts.map((attempt, index) => (
          <div key={index} className="attempt-row">
            {attempt.guess.split('').map((letter, i) => {
              const feedbackClass = 
                attempt.feedback[i] === 'G' ? 'green' : 
                attempt.feedback[i] === 'Y' ? 'yellow' : 
                'gray'
              
              return (
                <div key={i} className={`letter-cell ${feedbackClass}`}>
                  {letter}
                </div>
              )
            })}
            <div className="attempt-info">
              Tour {index + 1} - {attempt.remainingCandidates} candidats restants
            </div>
          </div>
        ))}
        
        {/* Lignes vides pour atteindre 6 lignes */}
        {Array.from({ length: Math.max(0, maxRows - attempts.length) }).map((_, index) => (
          <div key={`empty-${index}`} className="attempt-row empty">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="letter-cell empty"></div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

export default WordleGrid

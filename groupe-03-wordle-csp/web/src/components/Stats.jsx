import '../styles/Stats.css'

function Stats({ stats, attemptsCount }) {
  return (
    <div className="stats">
      <h3>📊 Statistiques</h3>
      <div className="stats-grid">
        <div className="stat-item">
          <span className="stat-label">Solveur:</span>
          <span className="stat-value">{stats.solverType || 'filtering'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Dictionnaire:</span>
          <span className="stat-value">{stats.totalWords || '-'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Candidats restants:</span>
          <span className="stat-value">{stats.candidatesCount || '-'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Tentatives:</span>
          <span className="stat-value">{attemptsCount}</span>
        </div>
        {stats.solved && (
          <div className="stat-item success">
            <span className="stat-label">Statut:</span>
            <span className="stat-value">✅ Résolu</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default Stats

import React from 'react'
import clsx from 'clsx'

const Sidebar = ({ 
  history = [], 
  onHistoryClick = ()=>{}, 
  onClearHistory = ()=>{}, 
  onLogout = ()=>{},
  isOpen = true 
}) => {
  return (
    <aside className={clsx('sidebar', { open: isOpen })} aria-hidden={!isOpen}>
      <div className="sidebar-header">
        <h2>Histórico</h2>
      </div>
      <ul className="history-list">
        {history.length === 0 && <p className="small">Nenhuma pesquisa ainda</p>}
        {history.map((it, idx) => (
          <li key={idx} className="history-item" onClick={() => onHistoryClick(it)} title={it.pergunta || it.query}>
            {it.pergunta || it.query}
          </li>
        ))}
      </ul>
      <div style={{marginTop:'auto', paddingTop: '16px', borderTop: '1px solid var(--border-color)'}}>
        <button className="clear-history-btn" onClick={onClearHistory} style={{marginBottom: '8px'}}>
          Limpar histórico
        </button>
        <button className="clear-history-btn logout-btn" onClick={onLogout}>
          Sair (Logout)
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
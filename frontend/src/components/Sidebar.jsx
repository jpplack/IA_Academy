import React from 'react'
import clsx from 'clsx'


const Sidebar = ({ history = [], onHistoryClick = ()=>{}, isOpen = true, setOpen = ()=>{} }) => {
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


<div style={{marginTop:'auto'}}>
<button className="btn btn-secondary" onClick={() => { localStorage.removeItem('ai_history'); window.location.reload(); }} style={{width:'100%'}}>Limpar histórico</button>
</div>
</aside>
)
}


export default Sidebar
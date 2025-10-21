import React, { useCallback } from 'react'
import { SendIcon } from './Icons'


const InputBar = ({ query, setQuery, onSearch, isLoading }) => {
const handleKey = useCallback((e) => { if (e.key === 'Enter') onSearch() }, [onSearch])
return (
<div className="input-bar-container">
<div className="input-bar">
<input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={handleKey} placeholder="Digite sua pergunta e pressione Enter..." />
<button onClick={onSearch} disabled={isLoading} aria-label="Enviar">
<SendIcon />
</button>
</div>
<div className="footer-text">AI Academy — Pesquisas rápidas e precisas</div>
</div>
)
}


export default InputBar
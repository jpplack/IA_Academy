import React, { useEffect, useRef, useState } from 'react'


const Typing = ({ text, speed=18 }) => {
const [display, setDisplay] = useState('')
const iRef = useRef(0)
useEffect(() => {
setDisplay('')
iRef.current = 0
if(!text) return
const handle = setInterval(() => {
iRef.current += 1
setDisplay(text.slice(0, iRef.current))
if(iRef.current >= text.length) clearInterval(handle)
}, speed)
return () => clearInterval(handle)
}, [text, speed])
return <p style={{whiteSpace:'pre-wrap'}}>{display}</p>
}


const ChatWindow = ({ isLoading, error, activeSearch }) => {
const scrollRef = useRef(null)
useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [activeSearch, isLoading])


if (isLoading) return <div className="chat-window"><div className="loader" /></div>
if (error) return <div className="chat-window"><div className="error-message">{error}</div></div>
if (!activeSearch || !activeSearch.resumo_ia) return (
<div className="chat-window">
<div className="welcome-message">
<h1>Bem-vindo ao AI Academy</h1>
<p>Digite sua pergunta no campo abaixo e receba um resumo unificado das principais fontes acadêmicas.</p>
</div>
</div>
)


return (
<div className="chat-window" ref={scrollRef}>
<div className="response-container">
<section className="summary-section">
<h2>Resumo da IA</h2>
<Typing text={activeSearch.resumo_ia} speed={10} />
</section>


<section className="results-section">
<h3>Resultados</h3>
{activeSearch.resultados && activeSearch.resultados.map((res, idx) => (
<div className="result-item" key={idx}>
<h4>[{res.fonte}] 
  {res.url ? (
    <a href={res.url} target="_blank" rel="noreferrer" style={{color: 'inherit'}}>
      {res.titulo}
    </a>
  ) : (
    res.titulo
  )}
</h4>
<p className="small">{res.authors ? (res.authors.join(', ')) : ''}</p>
<p>{res.resumo}</p>
{<res className="url"></res> && <a href={res.url} target="_blank" rel="noreferrer">Abrir fonte</a>}
</div>
))}
</section>
</div>
</div>
)
}


export default ChatWindow
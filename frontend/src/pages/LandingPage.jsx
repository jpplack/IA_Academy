import React from 'react'
import { Link } from 'react-router-dom'
import './LandingPage.css'
import { LogoIcon } from '../components/Icons'


export const LandingPage = () => {
return (
<div className="landing-container container-center">
<header className="landing-header">
<div className="logo">
<LogoIcon />
<span style={{marginLeft:12,fontWeight:800}}>AI Academy</span>
</div>
<nav>
<a href="#sobre">Sobre</a>
<a href="#funcionalidades">Funcionalidades</a>
</nav>
<div className="auth-buttons">
<button className="btn btn-secondary">Login</button>
<Link to="/lab" className="btn btn-primary" style={{marginLeft:12}}>Comece a pesquisar</Link>
</div>
</header>


<main className="hero">
<h1 className="hero-title">PESQUISA CIENTÍFICA, <span className="gradient-text">REINVENTADA.</span></h1>
<p className="hero-subtitle">Sua IA assistente que lê, resume e conecta artigos científicos para você.</p>
<Link to="/lab" className="cta-button">COMECE A PESQUISAR AGORA ▸</Link>
</main>
</div>
)
}
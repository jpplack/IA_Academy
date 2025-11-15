import React from 'react'
import { Link } from 'react-router-dom'
import './LandingPage.css'
import logoImage from '../assets/Logo_AI.png' 

export const LandingPage = () => {
  return (
    <div className="landing-container container-center">
      <header className="landing-header">
        <div className="logo">
          <img 
            src={logoImage} 
            alt="AI Academy Logo" 
            style={{ height: '90px' }}
          />
        </div>
        <nav>
          <Link to="/sobre">Sobre</Link>
          <Link to="/funcionalidades">Funcionalidades</Link>
        </nav>
        <div className="auth-buttons">
          <Link to="/login" className="btn btn-secondary">Login</Link>
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
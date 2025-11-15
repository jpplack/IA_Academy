import React from 'react';
import './StaticPage.css';

const FuncionalidadesPage = () => {
  return (
    <div className="static-page-container">
      <div className="static-page-content">
        <h1>Funcionalidades</h1>
        <ul>
          <li><strong>Busca Paralela e Rápida:</strong> Consultas assíncronas simultâneas em múltiplas fontes de dados.</li>
          <li><strong>Inteligência com Google Gemini:</strong> Otimização de perguntas e geração de resumos agregados.</li>
          <li><strong>Histórico de Pesquisas:</strong> Salva suas buscas para que você nunca perca uma linha de pesquisa.</li>
          <li><strong>Arquitetura Moderna:</strong> Backend robusto em FastAPI e Frontend reativo com React + Vite.</li>
        </ul>
        <a href="/" className="cta-button">Voltar ao Início</a>
      </div>
    </div>
  );
};

export default FuncionalidadesPage;
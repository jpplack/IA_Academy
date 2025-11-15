import React from 'react';
import './StaticPage.css';

const SobrePage = () => {
  return (
    <div className="static-page-container">
      <div className="static-page-content">
        <h1>Sobre o AI Academy</h1>
        <p>
          O <strong>AI Academy</strong> é um assistente de pesquisa inteligente, projetado para transformar a maneira como estudantes e profissionais de tecnologia interagem com o conhecimento acadêmico.
        </p>
        <p>
          Nossa plataforma realiza buscas paralelas em múltiplas fontes de alta credibilidade (IEEE Xplore, Semantic Scholar, arXiv, Wikipedia, etc.) e utiliza o poder do Google Gemini para otimizar as perguntas e gerar resumos coesos em português, tornando a pesquisa mais rápida, intuitiva e eficiente.
        </p>
        <a href="/" className="cta-button">Voltar ao Início</a>
      </div>
    </div>
  );
};

export default SobrePage;
# 🎓 AI Academy: Pesquisa Acadêmica Potencializada por IA

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-blue)]()
[![Powered by FastAPI](https://img.shields.io/badge/backend-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Frontend in React + Vite](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)](https://vitejs.dev/)
[![AI by Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)](https://ai.google.dev/)

---

## 📘 Sobre o Projeto

🇧🇷 O **AI Academy** é um assistente de pesquisa inteligente, projetado para transformar a maneira como estudantes e profissionais de tecnologia interagem com o conhecimento acadêmico. A plataforma realiza buscas paralelas em múltiplas fontes de alta credibilidade (IEEE Xplore, Semantic Scholar, arXiv, Wikipedia, etc.) e utiliza o poder do Google Gemini para otimizar as perguntas e gerar resumos coesos em português, tornando a pesquisa mais rápida, intuitiva e eficiente.

🇺🇸 **AI Academy** is an intelligent research assistant designed to transform how technology students and professionals interact with academic knowledge. The platform performs parallel searches across multiple high-credibility sources (IEEE Xplore, Semantic Scholar, arXiv, Wikipedia, etc.) and leverages the power of Google Gemini to optimize queries and generate cohesive summaries in Portuguese, making research faster, more intuitive, and more efficient.

---

## ✨ Funcionalidades

- 🚀 **Busca Paralela e Rápida:** Consultas assíncronas simultâneas em até 6 fontes de dados, incluindo IEEE Xplore, Semantic Scholar, arXiv e Wikipedia.
- 🧠 **Inteligência com Google Gemini:**
    - **Otimização de Query:** Uma "válvula de escape" que reinterpreta perguntas em linguagem natural e as transforma em termos de busca técnicos para melhores resultados.
    - **Resumos Agregados:** A IA lê os artigos encontrados e gera um resumo único e coeso em português, respondendo diretamente à pergunta do usuário.
- 📚 **Histórico de Pesquisas:** Todas as buscas são salvas localmente, permitindo que o usuário revisite e explore resultados anteriores.
- ⚙️ **Arquitetura Moderna:**
    - Backend robusto e assíncrono em **FastAPI**.
    - Frontend reativo e performático construído com **React + Vite**.
- 🔒 **Foco em Privacidade:** O sistema não requer contas de usuário e opera de forma segura, utilizando chaves de API armazenadas em um arquivo `.env` local.

---

## 🚀 Como Executar Localmente

### 🔧 Pré-requisitos
- Python 3.11+
- Node.js 18+ (LTS)
- Git (opcional, para clonar o repositório)

### 🧠 Backend (FastAPI + Gemini)

1.  **Acesse a pasta do backend:**
    ```bash
    cd AI_ACADEMY
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Criar o ambiente
    python -m venv venv
    
    # Ativar no Windows (PowerShell)
    .\venv\Scripts\activate
    
    # Ativar no Linux/macOS
    # source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Inicie a API:**
    ```bash
    # Para rodar na porta 8080
    uvicorn main:app --reload --port 8080
    ```
    A API estará disponível em: `http://localhost:8080`

### 💻 Frontend (React + Vite)

1.  **Abra um novo terminal** e acesse a pasta do frontend:
    ```bash
    cd AI_ACADEMY/frontend
    ```

2.  **Instale os pacotes (apenas na primeira vez):**
    ```bash
    npm install
    ```

3.  **Inicie o servidor de desenvolvimento:**
    ```bash
    npm run dev
    ```
    A aplicação estará disponível em: `http://localhost:5173` (ou outra porta indicada pelo Vite).

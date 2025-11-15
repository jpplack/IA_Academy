# 🎓 AI Academy: Assistente de Pesquisa com IA

[](https://www.google.com/search?q=LICENSE)
[](https://www.google.com/search?q=)
[](https://fastapi.tiangolo.com/)
[](https://vitejs.dev/)
[](https://ai.google.dev/)
[](https://www.sqlalchemy.org/)
[](https://passlib.readthedocs.io/en/stable/lib/passlib.hash.argon2.html)

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-blue)](https://github.com/galdino013/AI_ACADEMY)
[![Backend](https://img.shields.io/badge/backend-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)](https://vitejs.dev/)
[![AI](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)](https://ai.google.dev/)

-----

## 📘 Sobre o Projeto

O **AI Academy** é um assistente de pesquisa inteligente e seguro, projetado para transformar a maneira como estudantes e profissionais interagem com o conhecimento acadêmico. A plataforma agora é um sistema multiusuário completo, onde cada usuário possui uma conta segura e um histórico de pesquisa privado.

A plataforma realiza buscas paralelas em múltiplas fontes de alta credibilidade (IEEE Xplore, Semantic Scholar, arXiv, etc.) e utiliza o poder do Google Gemini para otimizar as perguntas e gerar resumos coesos em português, tornando a pesquisa mais rápida, intuitiva e eficiente.

-----

## 🚀 Demo Online

A aplicação está hospedada e pode ser acessada publicamente.

**Link da Aplicação:** **[https://aiacademy2025.netlify.app/](https://aiacademy2025.netlify.app/)**

> **Nota para a Banca:**
>
>   * O backend está hospedado no Render.com (plano gratuito) e **"dorme" após 15 minutos de inatividade**.
>   * A primeira requisição (login ou registro) pode demorar de **30 a 60 segundos** para "acordar" o servidor. Após isso, o site funcionará com velocidade normal.
>   * O banco de dados do deploy é "efêmero", então os usuários de teste são limpos periodicamente. Sinta-se à vontade para criar uma nova conta.

-----

## 📘 Sobre o Projeto

O **AI Academy** é um assistente de pesquisa inteligente e seguro, projetado para transformar a maneira como estudantes e profissionais interagem com o conhecimento acadêmico. A plataforma agora é um sistema multiusuário completo, onde cada usuário possui uma conta segura e um histórico de pesquisa privado.

A plataforma realiza buscas paralelas em múltiplas fontes de alta credibilidade (IEEE Xplore, Semantic Scholar, arXiv, etc.) e utiliza o poder do Google Gemini para otimizar as perguntas e gerar resumos coesos em português, tornando a pesquisa mais rápida, intuitiva e eficiente.

-----

## ✨ Funcionalidades Principais

  * 🚀 **Autenticação Segura (JWT):** Sistema completo de registro (`/users/register`) e login (`/token`). As senhas são protegidas no banco de dados usando hashing **Argon2**, e as sessões são gerenciadas por **JSON Web Tokens (JWT)**.
  * 🔒 **Rotas Protegidas:** Os endpoints de pesquisa (`/perguntar`) e histórico (`/historico`) são totalmente protegidos. Apenas usuários autenticados com um token válido podem acessá-los.
  * 📚 **Histórico por Usuário:** Cada usuário tem seu próprio histórico de pesquisa privado, persistido em um banco de dados **SQLite** e vinculado ao seu `user_id`. O histórico de um usuário não é visível para nenhum outro.
  * 🧠 **Inteligência com Google Gemini:**
      * **Otimização de Query:** Se a busca inicial falhar, a IA reinterpreta a pergunta e a transforma em termos técnicos para uma segunda tentativa.
      * **Resumos Agregados:** A IA lê os artigos encontrados e gera um resumo único e coeso em português.
  * ⚙️ **Arquitetura Moderna:**
      * **Backend:** Robusto e assíncrono em **FastAPI**, com **SQLAlchemy** gerenciando o banco de dados **SQLite**.
      * **Frontend:** Reativo e performático construído com **React + Vite**, utilizando **React Context** para gerenciamento de estado de autenticação.
  * ⚡ **Busca Paralela:** Consultas assíncronas simultâneas em múltiplas fontes de dados (IEEE, Semantic Scholar, arXiv, PubMed, SciELO, Wikipedia).

-----

## 🛠️ Stack de Tecnologias

  * **Backend:** Python 3.11+, FastAPI, SQLAlchemy (SQLite), Passlib (Argon2), PyJWT (python-jose).
  * **Frontend:** Node.js 18+, React, Vite, React Router, Axios.
  * **APIs de IA:** Google Gemini (SDK `google-genai`), OpenAI (como fallback).

-----

## 📂 Estrutura do Projeto

O projeto é um "monorepo" contendo duas aplicações principais:

```
/AI_ACADEMY/
├── .gitignore
├── README.md
├── venv/
├── backend/
│   ├── __init__.py
│   ├── main.py       (API Principal FastAPI)
│   ├── crud.py       (Lógica do Banco de Dados)
│   ├── models.py     (Tabelas do SQLAlchemy)
│   ├── schemas.py    (Modelos Pydantic)
│   ├── security.py   (Hashing e Lógica JWT)
│   ├── database.py   (Conexão com o DB)
│   ├── requirements.txt
│   └── .env          (Chaves de API)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── .env          (URL da API)
    ├── public/
    │   ├── _redirects  (Regra do Netlify)
    │   └── favicon.png
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── assets/
        ├── components/
        ├── context/
        └── pages/
```

-----

## 🚀 Como Executar Localmente

### 🔧 Pré-requisitos

  * Python 3.11+
  * Node.js 18+ (LTS)
  * Git (opcional, para clonar)

### 1\. Configuração do Ambiente

1.  Clone o repositório:

    ```bash
    git clone https://github.com/galdino013/AI_ACADEMY.git
    cd AI_ACADEMY
    ```

2.  Crie e ative um ambiente virtual (venv) na pasta raiz:

    ```bash
    # Criar o ambiente
    python -m venv venv

    # Ativar no Windows (PowerShell)
    .\venv\Scripts\activate

    # Ativar no Linux/macOS
    # source venv/bin/activate
    ```

### 2\. 🧠 Backend (FastAPI + SQLAlchemy)

1.  **Instale as dependências do backend:**
    *(O `requirements.txt` está dentro da pasta `backend/`)*

    ```bash
    pip install -r backend/requirements.txt
    ```

2.  **Configure o `.env` do backend:**
    Crie um arquivo chamado `.env` dentro da pasta `backend/`. Copie o conteúdo abaixo e **preencha com suas chaves de API válidas**.

    ```ini
    # Chave do Google (SDK google-genai)
    GEMINI_API_KEY=AIzaSy...

    # Chave da OpenAI (para fallback)
    OPENAI_API_KEY=sk-...

    # Chave do Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY=...

    # Chave do IEEE (Nota: a aprovação pode demorar)
    IEEE_API_KEY=...

    # Chave secreta para assinar os Tokens JWT
    SECRET_KEY=09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7

    # Configs de arquivos
    CACHE_FILE=search_cache.json
    CACHE_TTL_MINUTES=60
    API_CONCURRENCY=10
    ```

3.  **Inicie a API (a partir da pasta RAIZ):**
    Rode o Uvicorn apontando para o módulo `backend.main`.

    ```bash
    # Estando na pasta AI_ACADEMY (raiz)
    uvicorn backend.main:app --reload --port 8080
    ```

    A API estará disponível em: `http://localhost:8080`

### 3\. 💻 Frontend (React + Vite)

1.  **Abra um novo terminal** na pasta raiz `AI_ACADEMY`.

2.  **Acesse a pasta do frontend e instale os pacotes:**

    ```bash
    cd frontend
    npm install
    ```

3.  **Configure o `.env` do frontend:**
    Crie um arquivo `.env` na pasta `frontend/` que aponte para a sua API local:

    ```ini
    VITE_API_URL=http://localhost:8080
    ```

4.  **Inicie o servidor de desenvolvimento:**

    ```bash
    npm run dev
    ```

    A aplicação estará disponível em: `http://localhost:5173` (ou outra porta indicada pelo Vite).

# main.py — AI Academy (Versão Profissional e Completa)
# Autor: Igor Galdino (ajustes por ChatGPT)
# Data: 2025-10-13
#
# Requisitos (instalar no venv):
# pip install fastapi uvicorn httpx xmltodict aiofiles python-dotenv python-multipart pydantic rich google-generativeai openai unidecode

import os
import json
import asyncio
import logging
import traceback
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from functools import wraps

import httpx
import xmltodict
import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from rich.console import Console
from unidecode import unidecode

# Try optional imports for LLM fallbacks
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import openai
except Exception:
    openai = None

# --------------------------
# CONFIGURAÇÃO INICIAL
# --------------------------
load_dotenv()

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
IEEE_API_KEY = os.getenv("IEEE_API_KEY")
HISTORICO_FILE = os.getenv("HISTORICO_FILE", "search_history.json")
CACHE_FILE = os.getenv("CACHE_FILE", "search_cache.json")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "200"))
API_CONCURRENCY = int(os.getenv("API_CONCURRENCY", "10"))

DEFAULT_HEADERS = {"User-Agent": "AI-Academy-Bot/1.0 (+https://github.com/igor-galdino)"}

# Gemini model names confirmed (without -latest suffix)
GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]

# Configure LLM clients if keys are present
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # No network call here; we'll instantiate models when needed
    except Exception as e:
        print("⚠️ Aviso: Falha ao configurar Gemini:", e)
        genai = None

if openai and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print("⚠️ Aviso: Falha ao configurar OpenAI:", e)
        openai = None

# --------------------------
# LOGGING (rich)
# --------------------------
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)])
logger = logging.getLogger("ai_academy")
logger.setLevel(logging.INFO)

# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI(title="AI Academy - Backend Profissional", version="2025.10.13")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# MODELS (Pydantic)
# --------------------------
class PerguntaUsuario(BaseModel):
    texto: str = Field(..., min_length=1, description="Pergunta ou termo de pesquisa do usuário.")
    fontes: Optional[List[str]] = Field(None, description="Lista de fontes a consultar. Se omitido, todas serão usadas.")
    max_results: Optional[int] = Field(3, description="Número máximo de resultados por fonte (default=3).")
    resumir: Optional[bool] = Field(True, description="Se true, gera resumo via LLM (fallback disponível).")

class ArtigoResultado(BaseModel):
    source: str
    title: str
    url: Optional[str] = ""
    abstract: Optional[str] = ""
    authors: List[str] = Field(default_factory=list)

class RespostaBusca(BaseModel):
    resumo_ia: str
    resultados: List[ArtigoResultado]

# --------------------------
# CACHE (arquivo simples) e HISTÓRICO
# --------------------------
_cache_lock = asyncio.Lock()

async def _load_cache() -> Dict[str, Any]:
    try:
        async with _cache_lock:
            if not os.path.exists(CACHE_FILE):
                return {}
            async with aiofiles.open(CACHE_FILE, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content) if content else {}
    except Exception as e:
        logger.warning("Falha ao ler cache: %s", e)
        return {}

async def _save_cache(cache: Dict[str, Any]):
    try:
        async with _cache_lock:
            async with aiofiles.open(CACHE_FILE, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("Falha ao salvar cache: %s", e)

async def cache_get(key: str):
    cache = await _load_cache()
    item = cache.get(key)
    if not item:
        return None
    # TTL check
    ts = datetime.fromisoformat(item.get("_ts"))
    if datetime.utcnow() - ts > timedelta(minutes=CACHE_TTL_MINUTES):
        # expired
        cache.pop(key, None)
        await _save_cache(cache)
        return None
    return item.get("value")

async def cache_set(key: str, value: Any):
    cache = await _load_cache()
    cache[key] = {"_ts": datetime.utcnow().isoformat(), "value": value}
    await _save_cache(cache)

# HISTORY (persisted)
_history_lock = asyncio.Lock()

async def salvar_historico_item(item: Dict[str, Any]):
    try:
        async with _history_lock:
            data = []
            if os.path.exists(HISTORICO_FILE) and os.path.getsize(HISTORICO_FILE) > 0:
                async with aiofiles.open(HISTORICO_FILE, "r", encoding="utf-8") as f:
                    content = await f.read()
                    if content:
                        data = json.loads(content)
            data.insert(0, item)
            data = data[:MAX_HISTORY]
            async with aiofiles.open(HISTORICO_FILE, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning("Falha ao salvar histórico: %s", e)

async def ler_historico() -> List[Dict[str, Any]]:
    try:
        async with _history_lock:
            if not os.path.exists(HISTORICO_FILE) or os.path.getsize(HISTORICO_FILE) == 0:
                return []
            async with aiofiles.open(HISTORICO_FILE, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content) if content else []
    except Exception as e:
        logger.warning("Falha ao ler histórico: %s", e)
        return []

async def limpar_historico_async() -> bool:
    try:
        async with _history_lock:
            async with aiofiles.open(HISTORICO_FILE, "w", encoding="utf-8") as f:
                await f.write("[]")
        return True
    except Exception as e:
        logger.warning("Falha ao limpar histórico: %s", e)
        return False

# --------------------------
# METRICS
# --------------------------
METRICS = {"queries": 0, "cache_hits": 0, "last_query_time": None}

# --------------------------
# HELPERS: sanitize, retries, dedupe
# --------------------------
def normalize_query(q: str) -> str:
    if not q:
        return ""
    return unidecode(q).strip()

async def retry_async(fn, *args, retries: int = 2, delay: float = 0.8, backoff: float = 2.0, **kwargs):
    attempt = 0
    while True:
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            logger.debug("Retry %s/%s for %s due to %s", attempt, retries, fn.__name__, e)
            if attempt > retries:
                raise
            await asyncio.sleep(delay)
            delay *= backoff

def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        key = (r.get("source"), (r.get("title") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# --------------------------
# API SEARCH FUNCTIONS (async)
# Each returns List[Dict] with keys: source,title,url,abstract,authors
# --------------------------
async def buscar_semantic_scholar(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    if not SEMANTIC_API_KEY:
        return []
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": max_results, "fields": "title,abstract,authors,url"}
        headers = {**DEFAULT_HEADERS, "x-api-key": SEMANTIC_API_KEY}
        resp = await client.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        results = []
        for a in data:
            results.append({
                "source": "Semantic Scholar",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "abstract": a.get("abstract", ""),
                "authors": [auth.get("name") for auth in a.get("authors", []) if auth.get("name")]
            })
        return results
    except httpx.HTTPStatusError as e:
        logger.warning("Semantic Scholar HTTP error: %s", e)
        return []
    except Exception as e:
        logger.warning("Semantic Scholar error: %s", e)
        return []

async def buscar_ieee(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    if not IEEE_API_KEY:
        return []
    try:
        url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        params = {"querytext": query, "apikey": IEEE_API_KEY, "max_records": max_results}
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results = []
        for art in articles:
            # authors may be nested
            authors_field = []
            try:
                authors_field = [a.get("full_name") for a in art.get("authors", {}).get("authors", []) if a.get("full_name")]
            except Exception:
                pass
            results.append({
                "source": "IEEE Xplore",
                "title": art.get("title", ""),
                "url": art.get("html_url") or art.get("pdf_url") or "",
                "abstract": art.get("abstract", ""),
                "authors": authors_field
            })
        return results
    except httpx.HTTPStatusError as e:
        logger.warning("IEEE HTTP error: %s", e)
        return []
    except Exception as e:
        logger.warning("IEEE error: %s", e)
        return []

async def buscar_wikipedia(client: httpx.AsyncClient, query: str, max_results: int = 1) -> List[Dict[str, Any]]:
    try:
        # sanitize: remove trailing question marks and use PT wiki
        q = query.replace("?", "").strip()
        q = q.replace(" ", "_")
        url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{httpx.utils.quote(q)}"
        resp = await client.get(url, headers=DEFAULT_HEADERS, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        return [{
            "source": "Wikipedia",
            "title": data.get("title", query),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "abstract": data.get("extract", ""),
            "authors": []
        }]
    except httpx.HTTPStatusError as e:
        logger.debug("Wikipedia status error: %s", e)
        return []
    except Exception as e:
        logger.debug("Wikipedia error: %s", e)
        return []

async def buscar_arxiv(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url = "https://export.arxiv.org/api/query"
        params = {"search_query": f'all:"{query}"', "start": 0, "max_results": max_results}
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        data = xmltodict.parse(resp.text)
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            return []
        if not isinstance(entries, list):
            entries = [entries]
        results = []
        for e in entries:
            author_field = e.get("author", [])
            authors = []
            if isinstance(author_field, list):
                authors = [a.get("name") for a in author_field if isinstance(a, dict) and a.get("name")]
            elif isinstance(author_field, dict):
                authors = [author_field.get("name")]
            results.append({
                "source": "arXiv",
                "title": (e.get("title") or "").strip(),
                "url": (e.get("id") or ""),
                "abstract": (e.get("summary") or "").strip().replace("\n", " "),
                "authors": authors
            })
        return results
    except Exception as e:
        logger.debug("arXiv error: %s", e)
        return []

async def buscar_pubmed(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        resp = await client.get(url_search, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params_summary = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        resp2 = await client.get(url_summary, params=params_summary, headers=DEFAULT_HEADERS, timeout=15)
        resp2.raise_for_status()
        docs = resp2.json().get("result", {})
        results = []
        for id_, doc in docs.items():
            if id_ == "uids":
                continue
            authors = [a.get("name") for a in doc.get("authors", []) if a.get("name")] if isinstance(doc.get("authors"), list) else []
            results.append({
                "source": "PubMed",
                "title": doc.get("title", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_}/",
                "abstract": "",
                "authors": authors
            })
        return results
    except Exception as e:
        logger.debug("PubMed error: %s", e)
        return []

async def buscar_scielo(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url = "https://search.scielo.org/api/v1/search/"
        params = {"q": query, "count": max_results}
        # SciELO tends to block unknown clients; include User-Agent
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        results = []
        for a in data:
            try:
                title = a.get("title", [{}])[0].get("v", "") if a.get("title") else ""
            except Exception:
                title = ""
            try:
                abstract = a.get("abstract", [{}])[0].get("v", "") if a.get("abstract") else ""
            except Exception:
                abstract = ""
            authors = [author.get("v", "") for author in a.get("author", [])] if a.get("author") else []
            results.append({
                "source": "SciELO",
                "title": title,
                "url": a.get("url", ""),
                "abstract": abstract,
                "authors": authors
            })
        return results
    except httpx.HTTPStatusError as e:
        logger.warning("SciELO HTTP error: %s", e)
        return []
    except Exception as e:
        logger.debug("SciELO error: %s", e)
        return []

# --------------------------
# LLM Helpers: Gemini primary, OpenAI fallback, local fallback
# --------------------------
def _run_sync_genai_generate(model_name: str, prompt: str, max_tokens: int = 512) -> str:
    """Synchronous wrapper for Gemini generate_content via google.generativeai library."""
    if genai is None:
        raise RuntimeError("Gemini client não configurado")
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
        # Some clients return different structure
        try:
            return resp["candidates"][0]["content"][0]["text"].strip()
        except Exception:
            return ""
    except Exception as e:
        raise

def gerar_resumo_gemini_sync(pergunta: str, contexto: str) -> str:
    """Tenta gerar com os modelos GEMINI em ordem até funcionar, ou lança exceção."""
    if genai is None:
        raise RuntimeError("Gemini não disponível")
    for modelo in GEMINI_MODELS:
        try:
            logger.info("Tentando Gemini model %s", modelo)
            prompt = (
                f"Você é um assistente de pesquisa. Resuma de forma concisa e didática a seguinte informação respondendo à pergunta: '{pergunta}'. "
                f"Indique fontes entre colchetes. Seja claro e objetivo.\n\n{contexto}"
            )
            return _run_sync_genai_generate(modelo, prompt)
        except Exception as e:
            logger.warning("Falha com Gemini model %s: %s", modelo, e)
            continue
    raise RuntimeError("Nenhum modelo Gemini disponível ou todos falharam.")

def gerar_resumo_openai_sync(pergunta: str, contexto: str) -> str:
    """Fallback via OpenAI completion (se disponível)."""
    if openai is None:
        raise RuntimeError("OpenAI não disponível")
    try:
        prompt = (
            f"Resuma em português o conteúdo abaixo respondendo à pergunta: '{pergunta}'. "
            f"Cite as fontes entre colchetes. Seja sucinto.\n\n{contexto}"
        )
        # Usar GPT-4o-mini if available, otherwise gpt-4o or gpt-3.5-turbo
        model_name = "gpt-4o-mini" if "gpt-4o-mini" in getattr(openai, "__dict__", {}) else "gpt-3.5-turbo"
        # openai.ChatCompletion.create or openai.chat completions depending on SDK version:
        try:
            resp = openai.ChatCompletion.create(model=model_name, messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0.2)
            content = resp.choices[0].message.content
            return content.strip()
        except Exception:
            # fallback to Completion
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=512, temperature=0.2)
            return resp.choices[0].text.strip()
    except Exception as e:
        raise

def gerar_resumo_local(pergunta: str, contexto: str) -> str:
    """Resumo local e simples: concatena primeiros 200 chars de cada resultado e formata."""
    try:
        parts = []
        for seg in contexto.split("\n\n"):
            parts.append(seg.strip()[:300])
        joined = "\n\n".join(parts)
        summary = joined[:1500]
        footer = "\n\n(Resumo gerado localmente; habilite GEMINI ou OPENAI para resumos melhores.)"
        return summary + footer
    except Exception as e:
        logger.debug("Erro resumo local: %s", e)
        return "❌ Não foi possível gerar resumo."

async def gerar_resumo_por_fallback(pergunta: str, resultados: List[Dict[str, Any]]) -> str:
    contexto = "\n\n".join([f"[{r.get('source')}] {r.get('title')}\n{r.get('abstract','')}" for r in resultados if r.get("title")])
    # 1. Try Gemini synchronous in thread
    if genai is not None:
        try:
            resumo = await asyncio.to_thread(gerar_resumo_gemini_sync, pergunta, contexto)
            return resumo
        except Exception as e:
            logger.warning("Gemini falhou: %s", e)
    # 2. Try OpenAI
    if openai is not None:
        try:
            resumo = await asyncio.to_thread(gerar_resumo_openai_sync, pergunta, contexto)
            return resumo
        except Exception as e:
            logger.warning("OpenAI falhou: %s", e)
    # 3. Local fallback
    return gerar_resumo_local(pergunta, contexto)

# --------------------------
# ORQUESTADOR: junta tudo
# --------------------------
API_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY)

async def executar_buscas(query: str, fontes: Optional[List[str]], max_results: int) -> List[Dict[str, Any]]:
    fontes = [f.lower() for f in fontes] if fontes else ["semantic_scholar", "ieee", "wikipedia", "arxiv", "pubmed", "scielo"]
    key_cache = f"{query}::{'|'.join(sorted(fontes))}::{max_results}"
    cached = await cache_get(key_cache)
    if cached:
        METRICS["cache_hits"] += 1
        logger.info("Cache hit for query '%s' fontes=%s", query, fontes)
        return cached

    async with httpx.AsyncClient() as client:
        async with API_SEMAPHORE:
            tasks = []
            if "semantic_scholar" in fontes:
                tasks.append(retry_async(buscar_semantic_scholar, client, query, max_results))
            if "ieee" in fontes:
                tasks.append(retry_async(buscar_ieee, client, query, max_results))
            if "wikipedia" in fontes:
                tasks.append(retry_async(buscar_wikipedia, client, query, max_results))
            if "arxiv" in fontes:
                tasks.append(retry_async(buscar_arxiv, client, query, max_results))
            if "pubmed" in fontes:
                tasks.append(retry_async(buscar_pubmed, client, query, max_results))
            if "scielo" in fontes:
                tasks.append(retry_async(buscar_scielo, client, query, max_results))

            # If none selected, return empty
            if not tasks:
                return []

            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for r in results_lists:
                if isinstance(r, Exception):
                    logger.debug("Task exception: %s", r)
                    continue
                if isinstance(r, list):
                    results.extend(r)
            results = dedupe_results(results)
            # Normalize result fields and ensure url/title exist
            norm = []
            for r in results:
                norm.append({
                    "source": r.get("source", ""),
                    "title": r.get("title") or r.get("titulo") or "",
                    "url": r.get("url") or r.get("link") or "",
                    "abstract": r.get("abstract") or r.get("conteudo") or "",
                    "authors": r.get("authors") or r.get("autores") or []
                })
            await cache_set(key_cache, norm)
            return norm

# --------------------------
# ENDPOINTS
# --------------------------
@app.post("/perguntar", response_model=RespostaBusca)
async def perguntar(pergunta: PerguntaUsuario):
    q = normalize_query(pergunta.texto)
    if not q:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")
    METRICS["queries"] += 1
    METRICS["last_query_time"] = datetime.utcnow().isoformat()
    logger.info("[%s] Novo pedido: %s (fontes=%s max=%s)", METRICS["queries"], q, pergunta.fontes or "all", pergunta.max_results)

    # Executa buscas
    try:
        resultados = await executar_buscas(q, pergunta.fontes, pergunta.max_results)
    except Exception as e:
        logger.exception("Erro ao executar buscas: %s", e)
        resultados = []

    # Se nenhum resultado, tenta otimizar a query com LLM (apenas se resultados vazios)
    if not resultados:
        try:
            # Use Gemini optimize if available; else skip
            if genai is not None:
                prompt_opt = f"Analise a pergunta em português: '{pergunta.texto}'. Retorne uma query técnica curta em inglês para bases acadêmicas (apenas a query)."
                try:
                    termo_otimizado = await asyncio.to_thread(_run_sync_genai_generate, GEMINI_MODELS[0], prompt_opt)
                    termo_otimizado = termo_otimizado.strip().strip('\"')
                    logger.info("Query otimizada (Gemini): %s", termo_otimizado)
                    if termo_otimizado:
                        resultados = await executar_buscas(termo_otimizado, pergunta.fontes, pergunta.max_results)
                except Exception as e:
                    logger.debug("Gemini optimize falhou: %s", e)
            # Could add OpenAI optimize fallback here if desired
        except Exception as e:
            logger.debug("Erro na otimização: %s", e)

    if not resultados:
        # store empty result history with warning
        resumo = "❌ Não foi possível encontrar resultados nas fontes selecionadas."
        await salvar_historico_item({"pergunta": pergunta.texto, "resumo_ia": resumo, "resultados": [], "ts": datetime.utcnow().isoformat()})
        raise HTTPException(status_code=404, detail=resumo)

    # Gerar resumo com fallback LLMs (se pedido)
    resumo = ""
    if pergunta.resumir:
        try:
            resumo = await gerar_resumo_por_fallback(pergunta.texto, resultados)
        except Exception as e:
            logger.warning("Falha ao gerar resumo via LLMs: %s", e)
            resumo = gerar_resumo_local(pergunta.texto, "\n\n".join([r.get("abstract","") for r in resultados]))
    else:
        resumo = "Resumo não solicitado."

    # Salva histórico assincronamente (não bloquear a resposta)
    asyncio.create_task(salvar_historico_item({"pergunta": pergunta.texto, "resumo_ia": resumo, "resultados": resultados, "ts": datetime.utcnow().isoformat()}))

    # Converte para modelos pydantic (resposta bem formada)
    resultados_model = [ArtigoResultado(
        source=r.get("source", ""),
        title=r.get("title", ""),
        url=r.get("url", ""),
        abstract=r.get("abstract", ""),
        authors=r.get("authors", [])
    ) for r in resultados]

    return RespostaBusca(resumo_ia=resumo, resultados=resultados_model)

@app.get("/historico")
async def endpoint_ler_historico():
    return await ler_historico()

@app.delete("/historico")
async def endpoint_limpar_historico():
    ok = await limpar_historico_async()
    if not ok:
        raise HTTPException(status_code=500, detail="Erro ao limpar histórico.")
    return Response(status_code=204)

@app.get("/metrics")
async def get_metrics():
    return METRICS

@app.get("/")
async def root():
    return {"status": "AI Academy Backend (Profissional) ativo", "version": app.version, "metrics": METRICS}

# Serve a favicon to reduce 404 noise (opcional)
@app.get("/favicon.ico")
def favicon():
    # If you have a file, return FileResponse. For now, return 204.
    return Response(status_code=204)

# --------------------------
# RUN (apenas quando executado diretamente)
# --------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando AI Academy Backend (profissional) na porta 8080")
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)

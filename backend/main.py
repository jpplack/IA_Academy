# main.py — AI Academy
# Autor: Igor Galdino
# Data: 2025-10-13
# Versão: 2.6 (Histórico por Usuário no DB)

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
from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from rich.console import Console
from unidecode import unidecode
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from .import models, crud, schemas, security
from pathlib import Path
from .database import engine, SessionLocal
from sqlalchemy.orm import Session

try:
    from google import genai as google_genai_sdk
except Exception:
    google_genai_sdk = None
try:
    import openai
except Exception:
    openai = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
IEEE_API_KEY = os.getenv("IEEE_API_KEY")
CACHE_FILE = BASE_DIR / os.getenv("CACHE_FILE", "search_cache.json")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "200"))
API_CONCURRENCY = int(os.getenv("API_CONCURRENCY", "10")) 
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
GEMINI_MODELS = ["gemini-2.5-flash"]

if google_genai_sdk and GEMINI_API_KEY:
    try: genai_client = google_genai_sdk.Client()
    except Exception as e: print(f"⚠️ Aviso: Falha ao instanciar 'genai.Client()': {e}"); genai_client = None; google_genai_sdk = None
else: genai_client = None; google_genai_sdk = None
if openai and OPENAI_API_KEY: pass
else: openai = None

console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)])
logger = logging.getLogger("ai_academy")
logger.setLevel(logging.INFO)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Academy - Backend Profissional", version="2025.11.07 (v2.6)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Não foi possível validar as credenciais",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = security.verify_token(token, credentials_exception)
    user = crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

class PerguntaUsuario(BaseModel):
    texto: str = Field(..., min_length=1, description="Pergunta ou termo de pesquisa do usuário.")
    fontes: Optional[List[str]] = Field(None, description="Lista de fontes a consultar. Se omitido, todas serão usadas.")
    max_results: Optional[int] = Field(3, description="Número máximo de resultados por fonte (default=3).")
    resumir: Optional[bool] = Field(True, description="Se true, gera resumo via LLM (fallback disponível).")
class ArtigoResultado(BaseModel):
    source: str; title: str; url: Optional[str] = ""; abstract: Optional[str] = ""; authors: List[str] = Field(default_factory=list)
class RespostaBusca(BaseModel):
    resumo_ia: str; resultados: List[ArtigoResultado]

_cache_lock = asyncio.Lock()
async def _load_cache() -> Dict[str, Any]:
    try:
        async with _cache_lock:
            if not os.path.exists(CACHE_FILE): return {}
            async with aiofiles.open(CACHE_FILE, "r", encoding="utf-8") as f:
                content = await f.read(); return json.loads(content) if content else {}
    except Exception as e: logger.warning("Falha ao ler cache: %s", e); return {}
async def _save_cache(cache: Dict[str, Any]):
    try:
        async with _cache_lock:
            async with aiofiles.open(CACHE_FILE, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache, ensure_ascii=False, indent=2))
    except Exception as e: logger.warning("Falha ao salvar cache: %s", e)
async def cache_get(key: str):
    cache = await _load_cache(); item = cache.get(key)
    if not item: return None
    try:
        ts = datetime.fromisoformat(item.get("_ts"))
        if datetime.utcnow() - ts > timedelta(minutes=CACHE_TTL_MINUTES):
            cache.pop(key, None); await _save_cache(cache); return None
    except Exception: return None
    return item.get("value")
async def cache_set(key: str, value: Any):
    cache = await _load_cache(); cache[key] = {"_ts": datetime.utcnow().isoformat(), "value": value}; await _save_cache(cache)

METRICS = {"queries": 0, "cache_hits": 0, "last_query_time": None}

def normalize_query(q: str) -> str:
    if not q: return ""
    return unidecode(q).replace("?", "").strip()
async def retry_async(fn, *args, retries: int = 2, delay: float = 0.8, backoff: float = 2.0, **kwargs):
    attempt = 0
    while True:
        try: return await fn(*args, **kwargs)
        except Exception as e:
            attempt += 1; logger.debug("Retry %s/%s for %s due to %s", attempt, retries, fn.__name__, e)
            if attempt > retries: raise
            await asyncio.sleep(delay); delay *= backoff
def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for r in results:
        key = (r.get("source"), (r.get("title") or "").strip().lower())
        if not key[1]: continue
        if key in seen: continue
        seen.add(key); out.append(r)
    return out
async def buscar_semantic_scholar(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    if not SEMANTIC_API_KEY: return []
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"; params = {"query": query, "limit": max_results, "fields": "title,abstract,authors,url"}
        headers = {**DEFAULT_HEADERS, "x-api-key": SEMANTIC_API_KEY}; resp = await client.get(url, params=params, headers=headers, timeout=15); resp.raise_for_status()
        data = resp.json().get("data", []); results = []
        for a in data:
            results.append({"source": "Semantic Scholar", "title": a.get("title", ""), "url": a.get("url", ""), "abstract": a.get("abstract", ""), "authors": [auth.get("name") for auth in a.get("authors", []) if auth.get("name")]})
        return results
    except httpx.HTTPStatusError as e: logger.warning("Semantic Scholar HTTP error: %s", e); return []
    except Exception as e: logger.warning("Semantic Scholar error: %s", e); return []
async def buscar_ieee(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    if not IEEE_API_KEY: return []
    try:
        url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"; params = {"querytext": query, "apikey": IEEE_API_KEY, "max_records": max_results}
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15); resp.raise_for_status()
        articles = resp.json().get("articles", []); results = []
        for art in articles:
            authors_field = [];
            try: authors_field = [a.get("full_name") for a in art.get("authors", {}).get("authors", []) if a.get("full_name")]
            except Exception: pass
            results.append({"source": "IEEE Xplore", "title": art.get("title", ""), "url": art.get("html_url") or art.get("pdf_url") or "", "abstract": art.get("abstract", ""), "authors": authors_field})
        return results
    except httpx.HTTPStatusError as e: logger.warning("IEEE HTTP error: %s", e); return []
    except Exception as e: logger.warning("IEEE error: %s", e); return []
async def buscar_wikipedia(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Busca na Wikipedia usando a API de 'search' (inteligente), 
    não a API de 'summary' (título exato).
    """
    try:
        # 1. USA A API DE PESQUISA (SEARCH)
        url = "https://pt.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query, # Usa a query original (com "O que é...")
            "srlimit": max_results,
            "format": "json"
        }
        
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            logger.debug("Wikipedia search: Nenhum resultado para '%s'", query)
            return []

        results = []
        for item in search_results:
            page_id = item.get("pageid")
            # O 'snippet' é o resumo da busca, removemos o HTML
            snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace('</span>', "")
            
            results.append({
                "source": "Wikipedia",
                "title": item.get("title", ""),
                "url": f"https://pt.wikipedia.org/?curid={page_id}", # Link permanente
                "abstract": snippet + "...",
                "authors": [] # Wikipedia não tem autores por artigo
            })
        return results
    
    except httpx.HTTPStatusError as e:
        logger.debug("Wikipedia search API status error: %s", e)
        return []
    except Exception as e:
        logger.debug("Wikipedia search API error: %s", e)
        return []
async def buscar_arxiv(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url = "https://export.arxiv.org/api/query"; params = {"search_query": f'all:"{query}"', "start": 0, "max_results": max_results}
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15); resp.raise_for_status()
        data = xmltodict.parse(resp.text); entries = data.get("feed", {}).get("entry", [])
        if not entries: return []
        if not isinstance(entries, list): entries = [entries]
        results = []
        for e in entries:
            author_field = e.get("author", []); authors = []
            if isinstance(author_field, list): authors = [a.get("name") for a in author_field if isinstance(a, dict) and a.get("name")]
            elif isinstance(author_field, dict): authors = [author_field.get("name")]
            results.append({"source": "arXiv", "title": (e.get("title") or "").strip(), "url": (e.get("id") or ""), "abstract": (e.get("summary") or "").strip().replace("\n", " "), "authors": authors})
        return results
    except Exception as e: logger.debug("arXiv error: %s", e); return []
async def buscar_pubmed(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"; params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        resp = await client.get(url_search, params=params, headers=DEFAULT_HEADERS, timeout=15); resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids: return []
        url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"; params_summary = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        resp2 = await client.get(url_summary, params=params_summary, headers=DEFAULT_HEADERS, timeout=15); resp2.raise_for_status()
        docs = resp2.json().get("result", {}); results = []
        for id_, doc in docs.items():
            if id_ == "uids": continue
            authors = [a.get("name") for a in doc.get("authors", []) if a.get("name")] if isinstance(doc.get("authors"), list) else []
            results.append({"source": "PubMed", "title": doc.get("title", ""), "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_}/", "abstract": "", "authors": authors})
        return results
    except Exception as e: logger.debug("PubMed error: %s", e); return []
async def buscar_scielo(client: httpx.AsyncClient, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        url = "https://search.scielo.org/api/v1/search/"; params = {"q": query, "count": max_results}
        resp = await client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=15); resp.raise_for_status()
        data = resp.json().get("results", []); results = []
        for a in data:
            try: title = a.get("title", [{}])[0].get("v", "") if a.get("title") else ""
            except Exception: title = ""
            try: abstract = a.get("abstract", [{}])[0].get("v", "") if a.get("abstract") else ""
            except Exception: abstract = ""
            authors = [author.get("v", "") for author in a.get("author", [])] if a.get("author") else []
            results.append({"source": "SciELO", "title": title, "url": a.get("url", ""), "abstract": abstract, "authors": authors})
        return results
    except httpx.HTTPStatusError as e: logger.warning("SciELO HTTP error: %s", e); return []
    except Exception as e: logger.debug("SciELO error: %s", e); return []
def _run_sync_genai_generate(model_name: str, prompt: str) -> str:
    if genai_client is None or google_genai_sdk is None:
        raise RuntimeError("Gemini client (google-genai) não configurado")
    try:
        response = genai_client.models.generate_content(model=f"models/{model_name}", contents=prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Erro detalhado do _run_sync_genai_generate: {e}"); raise
async def gerar_resumo_gemini_async(pergunta: str, contexto: str) -> str:
    if google_genai_sdk is None: raise RuntimeError("Gemini não disponível")
    for modelo in GEMINI_MODELS:
        try:
            logger.info("Tentando Gemini model %s (Async, SDK Novo via to_thread)", modelo)
            prompt = (f"Você é um assistente de pesquisa. Resuma de forma concisa e didática a seguinte informação respondendo à pergunta: '{pergunta}'. " f"Indique fontes entre colchetes [Fonte]. Seja claro e objetivo.\n\n{contexto}")
            return await asyncio.to_thread(_run_sync_genai_generate, modelo, prompt)
        except Exception as e:
            logger.warning("Falha com Gemini model %s (Async, SDK Novo): %s", modelo, e); continue
    raise RuntimeError("Nenhum modelo Gemini disponível ou todos falharam.")
async def gerar_resumo_openai_async(pergunta: str, contexto: str) -> str:
    if openai is None or OPENAI_API_KEY is None:
        raise RuntimeError("OpenAI não disponível ou chave não configurada")
    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
         raise RuntimeError(f"Falha ao instanciar cliente AsyncOpenAI: {e}")
    try:
        logger.info("Tentando OpenAI model gpt-4o-mini (Async)")
        prompt = (f"Resuma em português o conteúdo abaixo respondendo à pergunta: '{pergunta}'. " f"Cite as fontes entre colchetes [Fonte]. Seja sucinto.\n\n{contexto}")
        model_name = "gpt-4o-mini" 
        resp = await client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0.2)
        content = resp.choices[0].message.content
        return content.strip()
    except Exception as e:
        logger.warning("Falha na chamada 'client.chat.completions.create' (Async): %s", e); raise
def gerar_resumo_local(pergunta: str, contexto: str) -> str:
    try:
        parts = [];
        for seg in contexto.split("\n\n"): parts.append(seg.strip()[:300])
        joined = "\n\n".join(parts); summary = joined[:1500]
        footer = "\n\n(Resumo gerado localmente; verifique suas chaves de API Gemini/OpenAI para resumos melhores.)"
        return summary + footer
    except Exception as e: logger.debug("Erro resumo local: %s", e); return "❌ Não foi possível gerar resumo."
async def gerar_resumo_por_fallback(pergunta: str, resultados: List[Dict[str, Any]]) -> str:
    contexto = "\n\n".join([f"[{r.get('source')}] {r.get('title')}\n{r.get('abstract','')}" for r in resultados if r.get("title")])
    if google_genai_sdk is not None:
        try:
            resumo = await gerar_resumo_gemini_async(pergunta, contexto)
            return resumo
        except Exception as e:
            logger.warning("Gemini falhou: %s", e)
    if openai is not None:
        try:
            resumo = await gerar_resumo_openai_async(pergunta, contexto)
            return resumo
        except Exception as e:
            logger.warning("OpenAI falhou: %s", e)
    logger.info("Ambas IAs falharam, usando fallback local."); return gerar_resumo_local(pergunta, contexto)
API_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY)
async def executar_buscas(query: str, fontes: Optional[List[str]], max_results: int) -> List[Dict[str, Any]]:
    fontes = [f.lower() for f in fontes] if fontes else ["semantic_scholar", "ieee", "wikipedia", "arxiv", "pubmed", "scielo"]
    key_cache = f"{query}::{'|'.join(sorted(fontes))}::{max_results}"
    cached = await cache_get(key_cache)
    if cached:
        METRICS["cache_hits"] += 1; logger.info("Cache hit for query '%s' fontes=%s", query, fontes); return cached
    async with httpx.AsyncClient() as client:
        async with API_SEMAPHORE:
            tasks = []
            if "semantic_scholar" in fontes: tasks.append(retry_async(buscar_semantic_scholar, client, query, max_results))
            if "ieee" in fontes: tasks.append(retry_async(buscar_ieee, client, query, max_results))
            if "wikipedia" in fontes: tasks.append(retry_async(buscar_wikipedia, client, query, max_results))
            if "arxiv" in fontes: tasks.append(retry_async(buscar_arxiv, client, query, max_results))
            if "pubmed" in fontes: tasks.append(retry_async(buscar_pubmed, client, query, max_results))
            if "scielo" in fontes: tasks.append(retry_async(buscar_scielo, client, query, max_results))
            if not tasks: return []
            results_lists = await asyncio.gather(*tasks, return_exceptions=True); results = []
            for r in results_lists:
                if isinstance(r, Exception): logger.debug("Task exception: %s", r); continue
                if isinstance(r, list): results.extend(r)
            results = dedupe_results(results); norm = []
            for r in results:
                norm.append({"source": r.get("source", ""), "title": r.get("title") or r.get("titulo") or "", "url": r.get("url") or r.get("link") or "", "abstract": r.get("abstract") or r.get("conteudo") or "", "authors": r.get("authors") or r.get("autores") or []})
            if norm: await cache_set(key_cache, norm)
            return norm

@app.post("/perguntar", response_model=RespostaBusca, tags=["Pesquisa (Protegido)"])
async def perguntar(
    pergunta: PerguntaUsuario,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = normalize_query(pergunta.texto)
    if not q: raise HTTPException(status_code=400, detail="Pergunta vazia.")
    
    logger.info("[%s] Novo pedido (usuário: %s): %s", METRICS["queries"], current_user.username, q)
    METRICS["queries"] += 1; METRICS["last_query_time"] = datetime.utcnow().isoformat()
    
    try:
        resultados = await executar_buscas(q, pergunta.fontes, pergunta.max_results)
    except Exception as e:
        logger.exception("Erro ao executar buscas: %s", e); resultados = []
    
    if not resultados:
        try:
            if google_genai_sdk is not None:
                prompt_opt = f"Analise a pergunta em português: '{pergunta.texto}'. Retorne uma query técnica curta em inglês para bases acadêmicas (apenas a query)."
                try:
                    termo_otimizado = await asyncio.to_thread(_run_sync_genai_generate, GEMINI_MODELS[0], prompt_opt)
                    termo_otimizado = termo_otimizado.strip().strip('\"')
                    if termo_otimizado and termo_otimizado.lower() != q.lower():
                        logger.info("Query otimizada (Gemini): %s", termo_otimizado)
                        resultados = await executar_buscas(termo_otimizado, pergunta.fontes, pergunta.max_results)
                except Exception as e: logger.debug("Gemini optimize falhou: %s", e)
        except Exception as e: logger.debug("Erro na otimização: %s", e)

    if not resultados:
        resumo = "❌ Não foi possível encontrar resultados nas fontes selecionadas."
        try:
            history_item_data = schemas.HistoryItemCreate(pergunta=pergunta.texto, resumo_ia=resumo, resultados=[])
            asyncio.create_task(asyncio.to_thread(crud.create_user_history_item, db=SessionLocal(), user_id=current_user.id, item=history_item_data))
        except Exception as e:
            logger.error(f"Falha ao salvar histórico de falha para user {current_user.id}: {e}")
            
        raise HTTPException(status_code=404, detail=resumo)

    resumo = ""
    if pergunta.resumir:
        try: resumo = await gerar_resumo_por_fallback(pergunta.texto, resultados)
        except Exception as e:
            logger.warning("Falha ao gerar resumo via LLMs: %s", e)
            resumo = gerar_resumo_local(pergunta.texto, "\n\n".join([r.get("abstract","") for r in resultados]))
    else: resumo = "Resumo não solicitado."
    try:
        history_item_data = schemas.HistoryItemCreate(
            pergunta=pergunta.texto, 
            resumo_ia=resumo, 
            resultados=resultados
        )
        asyncio.create_task(asyncio.to_thread(
            crud.create_user_history_item, 
            db=SessionLocal(),
            user_id=current_user.id, 
            item=history_item_data
        ))
    except Exception as e:
        logger.error(f"Falha ao salvar histórico de sucesso para user {current_user.id}: {e}")    
    resultados_model = [ArtigoResultado(source=r.get("source", ""), title=r.get("title", ""), url=r.get("url", ""), abstract=r.get("abstract", ""), authors=r.get("authors", [])) for r in resultados]
    return RespostaBusca(resumo_ia=resumo, resultados=resultados_model)

@app.get("/historico", response_model=List[schemas.HistoryItem], tags=["Pesquisa (Protegido)"])
async def endpoint_ler_historico(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
): 
    return crud.get_user_history(db=db, user_id=current_user.id)

@app.delete("/historico", tags=["Pesquisa (Protegido)"])
async def endpoint_limpar_historico(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ok = await asyncio.to_thread(crud.clear_user_history, db=db, user_id=current_user.id)
    if not ok: 
        raise HTTPException(status_code=500, detail="Erro ao limpar histórico.")
    return Response(status_code=204)

@app.get("/metrics", tags=["Admin"])
async def get_metrics(): return METRICS
@app.get("/", tags=["Admin"])
async def root(): return {"status": "AI Academy ativo", "version": app.version, "metrics": METRICS}
@app.get("/favicon.ico", include_in_schema=False)
def favicon(): return Response(status_code=204)
@app.post("/users/register", response_model=schemas.User, tags=["Autenticação"])
async def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username já cadastrado")
    
    new_user = await asyncio.to_thread(crud.create_user, db=db, user=user)
    return new_user

@app.post("/token", response_model=schemas.Token, tags=["Autenticação"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    is_password_correct = await asyncio.to_thread(
        security.verify_password, form_data.password, user.hashed_password
    )
    
    if not is_password_correct:
        raise HTTPException(
            status_code=401,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = security.create_access_token(
        data={"sub": user.username}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando AI Academy Backend v2.6")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
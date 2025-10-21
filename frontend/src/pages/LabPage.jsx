import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './LabPage.css'; // Garanta que este arquivo CSS exista e esteja correto

import Sidebar from '../components/Sidebar';
import ChatWindow from '../components/ChatWindow';
import InputBar from '../components/InputBar';
import { MenuIcon } from '../components/Icons';

function LabPage() {
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [activeSearch, setActiveSearch] = useState({ resumo_ia: '', resultados: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isSidebarOpen, setSidebarOpen] = useState(window.innerWidth > 800);

  const API_URL = 'http://127.0.0.1:8080'; // Centralizamos a URL da API

  const fetchHistory = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/historico`);
      setHistory(response.data);
    } catch (err) {
      console.error("Falha ao buscar histórico:", err);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleSearch = useCallback(async () => {
    if (query.length < 5) {
      setError('A pergunta precisa ter pelo menos 5 caracteres.');
      return;
    }
    setIsLoading(true);
    setError('');
    setActiveSearch({ resumo_ia: '', resultados: [] });
    if (window.innerWidth < 800) setSidebarOpen(false);

    try {
      const response = await axios.post(`${API_URL}/perguntar`, {
        texto: query
      });
      
      const newSearch = {
        pergunta: query,
        resumo_ia: response.data.resumo_ia,
        resultados: response.data.resultados,
      };
      
      setActiveSearch(newSearch);
      setHistory(prevHistory => [newSearch, ...prevHistory]); // Atualização otimista
      
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Ocorreu um erro ao buscar os dados. Verifique o console.';
      setError(errorMsg);
      console.error(err);
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  }, [query]);
  
  const handleHistoryClick = useCallback((item) => {
    setActiveSearch({
        resumo_ia: item.resumo_ia,
        resultados: item.resultados,
    });
    if (window.innerWidth < 800) setSidebarOpen(false);
  }, []);

  const handleClearHistory = useCallback(async () => {
    if (window.confirm("Você tem certeza que deseja limpar todo o histórico?")) {
      try {
        await axios.delete(`${API_URL}/historico`);
        setHistory([]);
        setActiveSearch({ resumo_ia: '', resultados: [] });
      } catch (err) {
        setError("Não foi possível limpar o histórico.");
        console.error("Falha ao limpar histórico:", err);
      }
    }
  }, []);

  return (
    <div className="app-container">
      <Sidebar 
        history={history}
        onHistoryClick={handleHistoryClick}
        onClearHistory={handleClearHistory}
        isOpen={isSidebarOpen}
      />

      <main className="main-content">
         <button className="menu-toggle" onClick={() => setSidebarOpen(!isSidebarOpen)}>
            <MenuIcon />
        </button>
        
        <ChatWindow 
          isLoading={isLoading}
          error={error}
          activeSearch={activeSearch}
        />
        
        <InputBar 
          query={query}
          setQuery={setQuery}
          onSearch={handleSearch}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}

export default LabPage;
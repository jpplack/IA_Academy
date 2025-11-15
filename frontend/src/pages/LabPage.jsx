import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './LabPage.css';
import Sidebar from '../components/Sidebar';
import ChatWindow from '../components/ChatWindow';
import InputBar from '../components/InputBar';
import { MenuIcon } from '../components/Icons';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

function LabPage() {
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [activeSearch, setActiveSearch] = useState({ resumo_ia: '', resultados: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isSidebarOpen, setSidebarOpen] = useState(window.innerWidth > 800);
  const { token, logout } = useAuth();
  const navigate = useNavigate();

  const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8080';
  const authHeaders = {
    headers: {
      Authorization: `Bearer ${token}`
    }
  };

  const fetchHistory = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/historico`, authHeaders);
      setHistory(response.data);
    } catch (err) {
      console.error("Falha ao buscar histórico:", err);
      if (err.response?.status === 401) {
        logout();
      }
    }
  }, [token, logout]);

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
      }, authHeaders); 
      
      const newSearch = {
        pergunta: query,
        resumo_ia: response.data.resumo_ia,
        resultados: response.data.resultados,
      };
      
      setActiveSearch(newSearch);
      setHistory(prevHistory => [newSearch, ...prevHistory]);
      
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Ocorreu um erro ao buscar.';
      setError(errorMsg);
      console.error(err);
      if (err.response?.status === 401) {
        logout();
      }
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  }, [query, token, logout]);
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
        await axios.delete(`${API_URL}/historico`, authHeaders);
        setHistory([]);
        setActiveSearch({ resumo_ia: '', resultados: [] });
      } catch (err) {
        setError("Não foi possível limpar o histórico.");
        console.error("Falha ao limpar histórico:", err);
        if (err.response?.status === 401) {
          logout();
        }
      }
    }
  }, [token, logout]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="app-container">
      <Sidebar 
        history={history}
        onHistoryClick={handleHistoryClick}
        onClearHistory={handleClearHistory}
        onLogout={handleLogout}
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
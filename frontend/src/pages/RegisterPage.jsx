import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './AuthPage.css';
import { useAuth } from '../context/AuthContext';

const RegisterPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { register, error } = useAuth(); 
  const navigate = useNavigate();
  const handleRegister = async (e) => {
    e.preventDefault();
    const success = await register(username, password);
    if (success) {
      navigate('/login');
    }
  };

  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleRegister}>
        <h1>Registro</h1>
        <p>Crie sua conta para começar.</p>
        {error && <div className="auth-error">{error}</div>}

        <div className="input-group">
          <label htmlFor="username">Usuário</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        
        <div className="input-group">
          <label htmlFor="password">Senha</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        
        <button type="submit" className="auth-button">Criar Conta</button>
        
        <p className="auth-switch">
          Já tem uma conta? <Link to="/login">Faça login</Link>
        </p>
      </form>
    </div>
  );
};

export default RegisterPage;
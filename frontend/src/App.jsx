// frontend/src/App.jsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import LabPage from './pages/LabPage';
import SobrePage from './pages/SobrePage'; // Importa a nova página
import FuncionalidadesPage from './pages/FuncionalidadesPage'; // Importa a nova página

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/lab" element={<LabPage />} />
      <Route path="/sobre" element={<SobrePage />} /> {/* Adiciona a nova rota */}
      <Route path="/funcionalidades" element={<FuncionalidadesPage />} /> {/* Adiciona a nova rota */}
    </Routes>
  );
}

export default App;
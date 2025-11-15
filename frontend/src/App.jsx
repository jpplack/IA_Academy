import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import LabPage from './pages/LabPage';
import SobrePage from './pages/SobrePage';
import FuncionalidadesPage from './pages/FuncionalidadesPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/sobre" element={<SobrePage />} />
      <Route path="/funcionalidades" element={<FuncionalidadesPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route element={<ProtectedRoute />}>
        <Route path="/lab" element={<LabPage />} />
      </Route>
      
    </Routes>
  );
}

export default App;
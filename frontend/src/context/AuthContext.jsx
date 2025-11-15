import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8080';

const AuthContext = createContext();

const actionTypes = {
  LOGIN_SUCCESS: 'LOGIN_SUCCESS',
  LOGOUT: 'LOGOUT',
  REGISTER_SUCCESS: 'REGISTER_SUCCESS',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
};

const initialState = {
  token: localStorage.getItem('token') || null,
  isAuthenticated: !!localStorage.getItem('token'),
  error: null,
};

const authReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.LOGIN_SUCCESS:
      return {
        ...state,
        token: action.payload.token,
        isAuthenticated: true,
        error: null,
      };
    case actionTypes.REGISTER_SUCCESS:
      return {
        ...state,
        error: null,
      };
    case actionTypes.LOGOUT:
      return {
        ...state,
        token: null,
        isAuthenticated: false,
        error: null,
      };
    case actionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload.error,
      };
    case actionTypes.CLEAR_ERROR:
       return {
        ...state,
        error: null,
      };
    default:
      return state;
  }
};

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);
  const login = async (username, password) => {
    try {
      dispatch({ type: actionTypes.CLEAR_ERROR });

      const params = new URLSearchParams();
      params.append('username', username);
      params.append('password', password);

      const response = await axios.post(`${API_URL}/token`, params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      const { access_token } = response.data;
      localStorage.setItem('token', access_token);
      dispatch({
        type: actionTypes.LOGIN_SUCCESS,
        payload: { token: access_token },
      });
      
      return true;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Erro ao fazer login.';
      dispatch({ type: actionTypes.SET_ERROR, payload: { error: errorMsg } });
      return false;
    }
  };

  const register = async (username, password) => {
    try {
      dispatch({ type: actionTypes.CLEAR_ERROR });
      
      await axios.post(`${API_URL}/users/register`, {
        username: username,
        password: password,
      });

      dispatch({ type: actionTypes.REGISTER_SUCCESS });
      return true;
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Erro ao registrar.';
      dispatch({ type: actionTypes.SET_ERROR, payload: { error: errorMsg } });
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    dispatch({ type: actionTypes.LOGOUT });
  };

  return (
    <AuthContext.Provider
      value={{
        ...state,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth deve ser usado dentro de um AuthProvider');
  }
  return context;
};
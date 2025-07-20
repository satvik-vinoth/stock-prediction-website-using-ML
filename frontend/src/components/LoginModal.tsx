'use client';
import React, { useState } from 'react';

interface Props {
  onClose: () => void;
  onSwitchToRegister: () => void;
  onLoginSuccess: () => void;
}

const LoginModal: React.FC<Props> = ({ onClose, onSwitchToRegister, onLoginSuccess }) => {
    const baseurl = process.env.NEXT_PUBLIC_API_BASE_URL
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) {
      setError('Email and Password are required.');
      return;
    }

    try {
      const res = await fetch(`${baseurl}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Login failed');
      localStorage.setItem('token', data.access_token); 
      onLoginSuccess();
      onClose();
    } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('An unknown error occurred.');
        }
    }
  };
  const handleClose = () => {
    onClose()
  };


  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white text-black p-6 rounded-lg w-96 shadow-lg">
        <div className="flex justify-between">
        <h2 className="text-xl font-bold mb-4">Login</h2>
        <button className="rounded pb-5 cursor-pointer" onClick={handleClose}>✕</button>

        </div>
        {error && <p className="text-red-500 mb-2">{error}</p>}
        <input
          type="email"
          placeholder="Email"
          className="w-full p-2 mb-3 border rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-2 mb-3 border rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button
          onClick={handleLogin}
          className="bg-[#39ff14] text-black w-full py-2 rounded hover:bg-green-600 cursor-pointer"
        >
          Login
        </button>
        <p className="text-sm mt-2 text-center">
          Don&apos;t have an account?{' '}
          <button onClick={onSwitchToRegister} className="text-blue-500 underline cursor-pointer">Register</button>
        </p>
      </div>
    </div>
  );
};

export default LoginModal;

'use client';
import React, { useState } from 'react';

interface Props {
  onClose: () => void;
  onSwitchToLogin: () => void;
}

const RegisterModal: React.FC<Props> = ({onClose, onSwitchToLogin }) => {
    const baseurl = process.env.NEXT_PUBLIC_API_BASE_URL
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const handleRegister = async () => {
    setError('');
    setSuccessMsg('');

    if (!email || !password || !confirm) {
      setError('All fields are required.');
      return;
    }

    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }

    try {
      const res = await fetch( `${baseurl}/register `, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Registration failed.');

      setSuccessMsg('Registration successful! Please log in.');
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
        <div className='flex justify-between'>
        <h2 className="text-xl font-bold mb-4">Register</h2>
        <button className="rounded pb-5 cursor-pointer" onClick={handleClose}>✕</button>
        </div>

        {error && <p className="text-red-500 mb-2">{error}</p>}
        {successMsg && <p className="text-green-600 mb-2">{successMsg}</p>}

        <input
          type="email"
          placeholder="Email"
          className="w-full p-2 mb-3 border rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={!!successMsg}
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-2 mb-3 border rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={!!successMsg}
        />
        <input
          type="password"
          placeholder="Confirm Password"
          className="w-full p-2 mb-3 border rounded"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          disabled={!!successMsg}
        />
        <button
          onClick={handleRegister}
          className="bg-[#39ff14] text-black w-full py-2 rounded hover:bg-green-600 cursor-pointer"
          disabled={!!successMsg}
        >
          Register
        </button>
        <p className="text-sm mt-2 text-center">
          Already have an account?{' '}
          <button onClick={onSwitchToLogin} className="text-blue-500 underline cursor-pointer">Log In</button>
        </p>
      </div>
    </div>
  );
};

export default RegisterModal;

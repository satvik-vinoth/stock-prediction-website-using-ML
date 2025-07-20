'use client';
import React, { useEffect } from 'react';
import { audiowide } from '@/lib/font';
import LoginModal from './LoginModal'; 
import RegisterModal from './RegisterModal';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';

const navItems = [
  { number: '01.', label: 'Home', href: '#home' },
  { number: '02.', label: 'Company', href: '#company' },
  { number: '03.', label: 'Prediction', href: '#prediction' },
  { number: '04.', label: 'Contact', href: '#contact' },
];


const Header: React.FC = () => {
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [token, setToken] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const tok = localStorage.getItem("token");
    if (tok){
      setToken(true)
    }
    
  },[]  )

  const handleLoginSuccess = () => {
    setToken(true);
    setShowLogin(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(false);
  };

  return (
    <>
    <header className="bg-[#1c3b35] text-[#a7ffeb] font-mono py-4 px-8 flex justify-between items-center fixed sticky top-0 z-50">

      <div className="flex items-center gap-2">
      <div className={`text-[#39ff14] rounded-md h-12 flex items-center justify-center text-2xl ${audiowide.className}`}>
        STOCK VISION
      </div>

      </div>

      <div className="hidden md:flex flex items-center gap-8">
        {navItems.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className="group transition"
          >
            <span className="text-[#39ff14] group-hover:text-white">{item.number}</span>
            <span className="text-gray-300 group-hover:text-white">{item.label}</span>
          </a>
        ))}
             {!token ? (
            <>
              <button
                onClick={() => setShowLogin(true)}
                className="ml-4 px-4 py-1 border border-[#39ff14] text-[#39ff14] rounded hover:bg-[#39ff14] hover:text-[#1c3b35] transform hover:scale-115 transition-all duration-150 ease-in-out cursor-pointer "
              >
                Log In
              </button>
              <button
                onClick={() => setShowRegister(true)}
                className="ml-2 px-4 py-1 border border-[#39ff14] text-[#39ff14] rounded hover:bg-[#39ff14] hover:text-[#1c3b35] transform hover:scale-115 transition-all cursor-pointer"
              >
                Sign Up
              </button>
            </>
          ) : (
            <button
              onClick={handleLogout}
              className="ml-4 px-4 py-1 border border-red-500 text-red-500 rounded hover:bg-red-600 hover:text-white transition-all transform hover:scale-115 transition-all cursor-pointer"
            >
              Logout
            </button>
          )}
      </div>
      <div className="md:hidden flex items-center">
          <button onClick={() => setMenuOpen(!menuOpen)} className='cursor-pointer'>
            {menuOpen ? <X size={24} color="#39ff14" /> : <Menu size={24} color="#39ff14" />}
          </button>
        </div>
    </header>
    {menuOpen && (
        <div className="md:hidden bg-[#1c3b35] text-[#a7ffeb] font-mono px-8 py-4 space-y-4 fixed top-16 w-full z-40 shadow-md">
          {navItems.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="block text-lg text-[#39ff14] hover:text-white"
              onClick={() => setMenuOpen(false)}
            >
              <span className="text-[#39ff14] group-hover:text-white">{item.number}</span>
              <span className="text-gray-300 group-hover:text-white">{item.label}</span>
            </a>
          ))}
          {!token ? (
            <>
              <button
                onClick={() => {
                  setShowLogin(true);
                  setMenuOpen(false);
                }}
                className="w-full px-4 py-2 border border-[#39ff14] text-[#39ff14] rounded hover:bg-[#39ff14] hover:text-[#1c3b35] cursor-pointer"
              >
                Log In
              </button>
              <button
                onClick={() => {
                  setShowRegister(true);
                  setMenuOpen(false);
                }}
                className="w-full px-4 py-2 border border-[#39ff14] text-[#39ff14] rounded hover:bg-[#39ff14] hover:text-[#1c3b35] cursor-pointer"
              >
                Sign Up
              </button>
            </>
          ) : (
            <button
              onClick={() => {
                handleLogout();
                setMenuOpen(false);
              }}
              className="w-full px-4 py-2 border border-red-500 text-red-500 rounded hover:bg-red-600 hover:text-white"
            >
              Logout
            </button>
          )}
        </div>
      )}
    {showLogin && (
        <LoginModal
          onClose={() => setShowLogin(false)}
          onSwitchToRegister={() => {
            setShowLogin(false);
            setShowRegister(true);
          }}
          onLoginSuccess={handleLoginSuccess}
        />
      )}

      {showRegister && (
        <RegisterModal
          onClose={() => setShowRegister(false)}
          onSwitchToLogin={() => {
            setShowRegister(false);
            setShowLogin(true);
          }}
        />
      )}
    </>
  );
};

export default Header;

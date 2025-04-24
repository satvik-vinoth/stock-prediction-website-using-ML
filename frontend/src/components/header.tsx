'use client';
import React from 'react';
import { audiowide } from '@/lib/font';

const navItems = [
  { number: '01.', label: 'Home', href: '#home' },
  { number: '02.', label: 'Company', href: '#company' },
  { number: '03.', label: 'Prediction', href: '#prediction' },
  { number: '04.', label: 'Contact', href: '#contact' },
];

const Header: React.FC = () => {
  return (
    <header className="bg-[#1c3b35] text-[#a7ffeb] font-mono py-4 px-8 flex justify-between items-center fixed sticky top-0 z-50">
      
      {/* Left: Neon Green "S" Logo */}
      <div className="flex items-center gap-2">
      <div className={`text-[#39ff14] rounded-md w-50 h-12 flex items-center justify-center  text-2xl ${audiowide.className}`}>
        STOCK VISION
      </div>

      </div>

      {/* Right: Navigation + Login */}
      <div className="flex items-center gap-8">
        {navItems.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className="group transition"
          >
            <span className="text-[#39ff14] group-hover:text-white">{item.number}</span>{' '}
            <span className="text-gray-300 group-hover:text-white">{item.label}</span>
          </a>
        ))}
        <a
          href="/login"
          className="ml-4 px-4 py-1 border border-[#39ff14] text-[#39ff14] rounded hover:bg-[#39ff14] hover:text-[#1c3b35] transition-all"
        >
          Log In
        </a>
      </div>
    </header>
  );
};

export default Header;

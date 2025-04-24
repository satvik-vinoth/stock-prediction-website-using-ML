'use client';
import React from 'react';
import { TextGenerateEffectDemo } from './ui/TextGenerateEffectDemo';
import { motion } from 'framer-motion';
import { orbitron } from '@/lib/font'; // or '@/lib/fonts' if placed there
import {inter} from '@/lib/font';

const FrontPage: React.FC = () => {
  return (
    <main className="relative mt-55 h-100">
      <div className='ml-30'>
        <h1 className={`text-4xl sm:text-5xl md:text-6xl font-bold text-[#39ff14] mb-4 ${orbitron.className}`}>
          <TextGenerateEffectDemo />
        </h1>
        
        <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
            className={`text-lg sm:text-xl text-gray-300 mt-4 `}
            >
            Get smart, <span className="text-[#39ff14] font-semibold">AI-powered</span> forecasts for your favorite stocks — all in real time.
            </motion.p>
        
      </div>
    </main>
  );
};

export default FrontPage;

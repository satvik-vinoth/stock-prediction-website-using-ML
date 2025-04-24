'use client';

import React, { useState, useEffect } from 'react';
import { PlaceholdersAndVanishInput } from '@/components/ui/placeholders-and-vanish-input';
import { orbitron } from '@/lib/font';

interface CompanySelectorProps {
  onCompanySelected: (symbol: string) => void;
}

const CompanySelector: React.FC<CompanySelectorProps> = ({ onCompanySelected }) => {
  const [selectedCompany, setSelectedCompany] = useState('AAPL');
  const [stockData, setStockData] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const placeholders = [
    "AAPL (Apple Inc.)",
    "GOOGL (Alphabet Inc.)",
    "TSLA (Tesla Inc.)",
    "MSFT (Microsoft Corp.)",
    "NVDA (NVIDIA Corp.)",
  ];

  useEffect(() => {
    fetchCompanyData("AAPL");
    onCompanySelected("AAPL");
  }, []);

  const fetchCompanyData = async (symbol: string) => {
    try {
      setLoading(true);
      const res = await fetch(`http://localhost:8000/stock/${symbol}`);
      if (!res.ok) throw new Error(`Failed to fetch ${symbol} data`);
      const data = await res.json();
      setStockData(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong fetching data");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedCompany(e.target.value.toUpperCase());
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setStockData(null);
    setLoading(true);

    await fetchCompanyData(selectedCompany);
    onCompanySelected(selectedCompany);
  };

  return (
    <section className="flex flex-col items-center text-white mt-20 px-4">
      <h2 className={`text-3xl font-bold text-[#39ff14] mb-10 text-center ${orbitron.className}`}>
        Choose a Company to Predict
      </h2>

      <div className="w-full max-w-xl">
        <PlaceholdersAndVanishInput
          placeholders={placeholders}
          onChange={handleChange}
          onSubmit={handleSubmit}
        />
      </div>

      {loading && <p className="mt-6 text-gray-400">Loading...</p>}
      {error && <p className="mt-6 text-red-400">{error}</p>}

      {stockData && (
        <div className="mt-10 w-full max-w-6xl overflow-x-auto rounded-lg shadow-lg border border-[#39ff14]">
          <table className="min-w-full text-sm text-left text-white bg-[#102b26]">
            <thead className="text-xs uppercase bg-[#1c3b35] text-[#39ff14]">
              <tr>
                <th className="px-4 py-2 text-center">Date</th>
                {Object.keys(Object.values(stockData)[0] as Record<string, number>).map((key) => (
                  <th key={key} className="px-2 py-2 whitespace-nowrap text-[#39ff14] uppercase text-xs text-center">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(stockData).map(([date, data]: any) => (
                <tr key={date} className="border-t border-[#39ff1455] hover:bg-[#18332e] transition text-center">
                  <td className="px-4 py-2 font-semibold text-[#39ff14]">{date.split('T')[0]}</td>
                  {Object.values(data).map((value: any, idx: number) => (
                    <td key={idx} className="px-4 py-2 whitespace-nowrap">{Number(value).toFixed(2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default CompanySelector;

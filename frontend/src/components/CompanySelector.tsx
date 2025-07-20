'use client';

import React, { useState, useEffect } from 'react';
import { PlaceholdersAndVanishInput } from '@/components/ui/placeholders-and-vanish-input';
import { orbitron } from '@/lib/font';

interface CompanySelectorProps {
  onCompanySelected: (symbol: string) => void;
}

const CompanySelector: React.FC<CompanySelectorProps> = ({ onCompanySelected }) => {

  interface StockEntry {
    Open: number;
    High: number;
    Low: number;
    Close: number;
    Volume: number;
    EMA_50?: number;
    MACD_12_26_9?: number;
    RSI_14?: number;
    OBV?: number;
    SMA_50?: number;
    [key: string]: number | undefined; 
  }
  
  type StockData = {
    [date: string]: StockEntry;
  };
  const baseurl = process.env.NEXT_PUBLIC_API_BASE_URL
  const [selectedCompany, setSelectedCompany] = useState('AAPL');
  const [stockData, setStockData] = useState<StockData| null>(null);
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
      const res = await fetch(`${baseurl}/stock/${symbol}`);
      if (!res.ok) throw new Error(`Failed to fetch ${symbol} data`);
      const data = await res.json();
      console.log(data)
      setStockData(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Something went wrong fetching data");
      }
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
                {Object.keys(Object.values(stockData)[0] as StockEntry).map((key) => (
                  <th key={key} className="px-2 py-2 whitespace-nowrap text-[#39ff14] uppercase text-xs text-center">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(stockData).map(([date, data]: [string, StockEntry]) => (
                <tr key={date} className="border-t border-[#39ff1455] hover:bg-[#18332e] transition text-center">
                  <td className="px-2 py-2 font-semibold text-[#39ff14]">{date.split('T')[0]}</td>
                  {Object.values(data).map((value, idx) => (
                    <td key={idx} className="px-4 py-2 whitespace-nowrap">
                      {typeof value === "number" ? value.toFixed(2) : "-"}
                    </td>
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

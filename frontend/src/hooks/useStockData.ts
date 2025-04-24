// hooks/useStockData.ts
import { useState } from 'react';

export const useStockData = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchStockData = async (symbol: string) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`http://localhost:8000/stock/${symbol}`);
      if (!res.ok) throw new Error("Stock not found");
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return { data, fetchStockData, loading, error };
};

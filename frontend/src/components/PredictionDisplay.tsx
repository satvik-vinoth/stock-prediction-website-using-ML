'use client';

import { useEffect, useState } from 'react';
import axios from 'axios';

interface PredictionResult {
  symbol: string;
  predicted_close: number;
}

export default function PredictionDisplay() {
  const [symbol, setSymbol] = useState('AAPL'); // Default
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchPrediction = async () => {
    try {
      setLoading(true);
      const res = await axios.get<PredictionResult>(
        `http://127.0.0.1:8000/predict/${symbol}`
      );
      setResult(res.data);
    } catch (err) {
      console.error('Prediction fetch failed', err);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
  }, [symbol]);

  return (
    <div className="bg-white shadow-lg rounded-xl p-6">
      <h2 className="text-xl font-bold text-green-800 mb-4">
        GRU Prediction Results
      </h2>

      <div className="flex items-center gap-4 mb-4">
        <label className="text-gray-600">Symbol:</label>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          className="border px-3 py-1 rounded text-sm outline-none"
          placeholder="e.g. AAPL"
        />
        <button
          onClick={fetchPrediction}
          className="bg-green-600 text-white px-4 py-1 rounded hover:bg-green-700"
        >
          Predict
        </button>
      </div>

      {loading && <p className="text-sm text-gray-500">Loading prediction...</p>}

      {!loading && result && (
        <div className="text-lg text-gray-800">
          <p>
            <strong>{result.symbol}</strong> → Predicted Close Price: <span className="text-green-700 font-semibold">${result.predicted_close}</span>
          </p>

          <div className="mt-6">
            {/* Chart.js will go here */}
            <p className="text-sm text-gray-500 italic">[Chart coming soon]</p>
          </div>
        </div>
      )}
    </div>
  );
}

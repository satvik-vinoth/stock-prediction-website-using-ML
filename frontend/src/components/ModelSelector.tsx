import { useState } from "react";
import { Button } from "@/components/ui/moving-border";
import { orbitron } from "@/lib/font";
import axios from "axios";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);


interface PredictionData {
    symbol: string;
    model: string;
    predicted_close: number;
    rmse: number;
    mape: number;
    recent_actual: number[][];  
    recent_predicted: number[];  
    error?: string;  
  }


  

const models = ["GRU", "LSTM", "Transformer"];

interface ModelSelectorProps {
  company: string;
}

export default function ModelSelector({ company }: ModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const baseurl = process.env.NEXT_PUBLIC_API_BASE_URL;
  const [login,setlogin] = useState(true);
  


  const handleModelClick = async (model: string) => {
    const token = localStorage.getItem("token");
    if (!token) {
      setlogin(false)
      return;
    }
    console.log(token)
    setSelectedModel(model);
    setPrediction(null);
    setLoading(true);


    try {
      const res = await axios.get(`${baseurl}/predict/${model.toLowerCase()}?symbol=${company}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setPrediction(res.data.prediction);
    } catch (err) {
      console.error("Prediction error:", err);
      setPrediction({
        symbol: "",
        model: "",
        predicted_close: 0,
        rmse: 0,
        mape: 0,
        recent_actual: [],
        recent_predicted: [],
        error: "Failed to load prediction.",
      });
      
    } finally {
      setLoading(false);
    }
  };

  const chartData = {
    labels: prediction?.recent_actual ? prediction.recent_actual.map((_, index: number) => `Day ${index + 1}`) : [],
    datasets: [
      {
        label: "Actual",
        data: prediction?.recent_actual ? prediction.recent_actual.map((val: number[]) => val[0]) : [],
        borderColor: "#FF5733",
        backgroundColor: "rgba(255, 87, 51, 0.2)",
        fill: false,
        tension: 0.1,
      },
      {
        label: "Predicted",
        data: prediction?.recent_predicted ? prediction.recent_predicted.map((val) => val) : [],
        borderColor: "#39ff14",
        backgroundColor: "rgba(255, 255, 255, 0.2)",
        fill: false,
        tension: 0.1,
      },
    ],
  };
  
  const options = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: "Actual vs Predicted Trend",
        color: "white",
      },
      tooltip: {
        mode: "index" as const,
        intersect: false,
      },
      legend: {
        display: true,
        labels: {
          color: "white",
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Days",
          color: "white",
        },
        ticks: {
          color: "white",
        },
      },
      y: {
        title: {
          display: true,
          text: "Price",
          color: "white",
        },
        ticks: {
          color: "white",
        },
      },
    },
  };
  
  
  return (
    <>
    <section className="flex flex-col items-center mt-20">
      <h2 className={`${orbitron.className} text-3xl text-[#39ff14] mb-8`}>Choose Your Model</h2>
  
      <div className="flex flex-wrap gap-6 justify-center">
        {models.map((model) => (
          <Button
            key={model}
            onClick={() => handleModelClick(model)}
            borderRadius="1.75rem"
            className="border-[#39ff14] text-white font-bold text-base tranform hover:scale-125 cursor-pointer"
            borderClassName="bg-[radial-gradient(#39ff14_40%,transparent_60%)]"
          >
            {model}
          </Button>
        ))}
      </div>
  
      <div className="mt-10 w-full max-w-xl text-center text-white">
        {loading && <p className="text-gray-400">Loading prediction...</p>}
  
        {!loading && prediction && !prediction.error && (
          <div className="mt-2 space-y-4">
            <p className="text-xl font-semibold text-[#39ff14]">
              Next Day Predicted Close: <span className="text-white">${prediction.predicted_close}</span>
            </p>
            
            <p className="text-sm text-gray-300">
              Model: {selectedModel} | Company: {company}
            </p>
  
            <div className="mt-6 border border-[#39ff14] rounded  "  style={{ width: '100%', height: '300px' ,margin:'0 auto'}}>
              {/* Display the chart here */}
              <Line data={chartData} options={options} />
            </div>
            <div className="flex gap-6 ml-9 mt-5">
                <div className="bg-[#46d90a] text-white p-4 rounded-lg  items-center w-60 text-center">
                    <p className="text-lg font-semibold">
                    RMSE: <span className="font-bold">{prediction.rmse}</span>
                    </p>
                </div>

                <div className="bg-[#46d90a] text-white p-4 rounded-lg text-center items-center w-60">
                    <p className="text-lg font-semibold">
                    MAPE: <span className="font-bold">{prediction.mape}%</span>
                    </p>
                </div>
                </div>

          </div>
        )}
  
        {prediction?.error && <p className="text-red-400 mt-4">{prediction.error}</p>}
      </div>
    </section>
    {!login && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 w-80 text-center">
          <h2 className="text-xl font-bold mb-4 text-red-600">Please Login</h2>
          <p className="text-gray-700 mb-6">You need to login to view the prediction.</p>
          <button
            onClick={() => setlogin(true)}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded"
          >
            OK
          </button>
        </div>
      </div>
    ) }
    </>
  );
}
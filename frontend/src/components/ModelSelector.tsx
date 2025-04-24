import { useState } from "react";
import { Button } from "@/components/ui/moving-border";
import { orbitron } from "@/lib/font";
import axios from "axios";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
import { color } from "chart.js/helpers";

// Register chart components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);


interface PredictionData {
    symbol: string;
    model: string;
    predicted_close: number;
    rmse: number;
    mape: number;
    recent_actual: number[][];  // Array of arrays of numbers
    recent_predicted: number[];  // Array of arrays of numbers
    error?: string;  // Make 'error' optional
  }
  

const models = ["GRU", "LSTM", "Transformer"];

interface ModelSelectorProps {
  company: string;
}

export default function ModelSelector({ company }: ModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);

  const handleModelClick = async (model: string) => {
    setSelectedModel(model);
    setPrediction(null);
    setLoading(true);

    try {
      const res = await axios.get(`http://localhost:8000/predict/${model.toLowerCase()}?symbol=${company}`);
      setPrediction(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
      setPrediction({ error: "Failed to load prediction." });
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
        color: "white"
      },
      tooltip: {
        mode: "index", // Corrected this to use a valid value from the list
        intersect: false,
      },
      legend:{
        display: true,
        labels:{
            color:"white"
          }
      }

    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Days",
          color: "white"
        },
        ticks: {
            color: "white", // Set the y-axis labels color to white
        },
      },
      y: {
        title: {
          display: true,
          text: "Price",
          color: "white"
        },
        ticks: {
            color: "white", // Set the y-axis labels color to white
        },
      }, 
    },

  };
  
  return (
    <section className="flex flex-col items-center mt-20">
      <h2 className={`${orbitron.className} text-3xl text-[#39ff14] mb-8`}>Choose Your Model</h2>
  
      <div className="flex flex-wrap gap-6 justify-center">
        {models.map((model) => (
          <Button
            key={model}
            onClick={() => handleModelClick(model)}
            borderRadius="1.75rem"
            className="border-[#39ff14] text-white font-bold text-base"
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
  );
  
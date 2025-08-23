
"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface PredictionResult {
  prediction: string;
  confidence: number;
  analysis?: {
    text_length: number;
    headline: string;
    content_preview: string;
  };
}

interface HistoryItem {
  headline: string;
  text: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
}

export default function Home() {
  const [headline, setHeadline] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // API URL - defaulting to localhost for development
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!headline.trim() && !text.trim()) {
      setError("Please provide either a headline or article text.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          headline: headline.trim(), 
          text: text.trim() 
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
      
      // Add to history
      const historyItem: HistoryItem = {
        headline: headline.trim(),
        text: text.trim(),
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: new Date()
      };
      
      setHistory((prev: HistoryItem[]) => [historyItem, ...prev.slice(0, 4)]);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to connect to the API. Please make sure the backend is running.";
      setError(errorMessage);
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setHeadline("");
    setText("");
    setResult(null);
    setError(null);
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 via-purple-800 to-pink-600 text-white">
      <div className="w-full max-w-4xl mx-auto p-8 rounded-3xl shadow-2xl bg-opacity-80 bg-gray-900 backdrop-blur-lg">
        <motion.h1
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 120 }}
          className="text-4xl font-extrabold mb-6 text-center tracking-tight bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 bg-clip-text text-transparent drop-shadow-lg"
        >
          🧠 Fake News Detector
        </motion.h1>

        <div className="mb-6 text-center">
          <span className="text-xs text-gray-400">
            Powered by <span className="font-bold text-pink-300">HuggingFace BART Large MNLI</span>
          </span>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <motion.input
            type="text"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            placeholder="Enter a news headline..."
            className="w-full border-2 border-pink-400 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 bg-gray-800 text-white placeholder:text-pink-300 shadow-md transition-all duration-300"
            whileFocus={{ scale: 1.02 }}
          />
          
          <motion.textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the news article text here..."
            rows={6}
            className="w-full border-2 border-purple-400 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-pink-400 bg-gray-800 text-white placeholder:text-purple-300 shadow-md transition-all duration-300 resize-vertical"
            whileFocus={{ scale: 1.02 }}
          />
          
          <div className="flex gap-4 flex-wrap">
            <motion.button
              type="submit"
              className="flex-1 py-3 px-6 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white font-bold rounded-xl shadow-lg hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              whileHover={{ scale: loading ? 1 : 1.05 }}
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                  Analyzing...
                </span>
              ) : (
                "Analyze News"
              )}
            </motion.button>
            
            <motion.button
              type="button"
              onClick={clearForm}
              className="px-6 py-3 bg-gray-600 hover:bg-gray-500 text-white font-bold rounded-xl transition-all duration-200"
              whileHover={{ scale: 1.05 }}
            >
              Clear
            </motion.button>
          </div>
        </form>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 rounded-xl bg-red-900/80 border border-red-500 text-red-200"
          >
            <h3 className="font-bold mb-2">Error:</h3>
            <p>{error}</p>
          </motion.div>
        )}

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.5 }}
              className="mt-8 p-6 rounded-2xl shadow-xl bg-gradient-to-br from-gray-800 via-purple-900 to-pink-800 border-2 border-blue-400"
            >
              <div className="text-center mb-4">
                <h2 className="text-2xl font-bold mb-2">Analysis Result</h2>
                <div className="flex justify-center items-center gap-4">
                  <span className="text-lg">Prediction:</span>
                  <span
                    className={`text-2xl font-bold px-4 py-2 rounded-lg ${
                      result.prediction.toLowerCase() === "real"
                        ? "bg-green-600 text-green-100"
                        : "bg-red-600 text-red-100"
                    }`}
                  >
                    {result.prediction.toUpperCase()}
                  </span>
                </div>
                <div className="mt-3">
                  <span className="text-lg mr-2">Confidence:</span>
                  <span className="text-xl font-mono text-yellow-300">
                    {result.confidence.toFixed(2)}%
                  </span>
                </div>
              </div>
              
              {result.analysis && (
                <div className="mt-4 p-4 bg-gray-800/50 rounded-lg">
                  <h3 className="font-bold mb-2">Analysis Details:</h3>
                  <div className="space-y-2 text-sm">
                    <p><span className="font-semibold">Text Length:</span> {result.analysis.text_length} characters</p>
                    {result.analysis.headline && (
                      <p><span className="font-semibold">Headline:</span> {result.analysis.headline}</p>
                    )}
                    {result.analysis.content_preview && (
                      <p><span className="font-semibold">Content Preview:</span> {result.analysis.content_preview}</p>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {history.length > 0 && (
          <div className="mt-10">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold text-pink-300">Recent Analysis History</h2>
              <button
                onClick={clearHistory}
                className="text-sm px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
              >
                Clear History
              </button>
            </div>
            <div className="space-y-3">
              {history.map((item, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="p-4 rounded-xl bg-gray-800/70 border border-purple-700"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      item.prediction.toLowerCase() === "real" 
                        ? "bg-green-600 text-green-100" 
                        : "bg-red-600 text-red-100"
                    }`}>
                      {item.prediction.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-400">
                      {item.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-sm space-y-1">
                    {item.headline && (
                      <p><span className="font-semibold text-blue-300">Headline:</span> {item.headline}</p>
                    )}
                    {item.text && (
                      <p><span className="font-semibold text-blue-300">Text:</span> {item.text.slice(0, 100)}...</p>
                    )}
                    <p><span className="font-semibold text-yellow-300">Confidence:</span> {item.confidence.toFixed(2)}%</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-10 text-center text-xs text-gray-400">
          <p className="mt-2 text-xs text-gray-500">
            Tip: Try different headlines and news articles to test the AI&apos;s detection capabilities!
          </p>
        </div>
      </div>
    </main>
  );
}

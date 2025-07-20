
"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";


export default function Home() {
  const [headline, setHeadline] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<null | { prediction: string; confidence: number }>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(process.env.NEXT_PUBLIC_API_URL + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ headline, text }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ prediction: "error", confidence: 0 });
    }
    setLoading(false);
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 via-purple-800 to-pink-600 text-white">
      <div className="w-full max-w-2xl mx-auto p-8 rounded-3xl shadow-2xl bg-opacity-80 bg-gray-900 backdrop-blur-lg">
        <motion.h1
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 120 }}
          className="text-4xl font-extrabold mb-6 text-center tracking-tight bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 bg-clip-text text-transparent drop-shadow-lg"
        >
          🧠 Fake News Detector
        </motion.h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <motion.input
            type="text"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            placeholder="Enter a catchy headline..."
            className="w-full border-2 border-pink-400 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 bg-gray-800 text-white placeholder:text-pink-300 shadow-md"
            required
            whileFocus={{ scale: 1.05 }}
          />
          <motion.textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the news article text here..."
            rows={6}
            className="w-full border-2 border-purple-400 p-3 rounded-xl focus:outline-none focus:ring-2 focus:ring-pink-400 bg-gray-800 text-white placeholder:text-purple-300 shadow-md"
            required
            whileFocus={{ scale: 1.05 }}
          />
          <motion.button
            type="submit"
            className="w-full py-3 px-6 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 text-white font-bold rounded-xl shadow-lg hover:scale-105 transition-transform duration-200"
            whileHover={{ scale: 1.07 }}
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>
                Predicting...
              </span>
            ) : (
              "Predict"
            )}
          </motion.button>
        </form>

        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.5 }}
              className="mt-8 p-6 rounded-2xl shadow-xl bg-gradient-to-br from-gray-800 via-purple-900 to-pink-800 border-2 border-blue-400 text-center"
            >
              {result.prediction === "error" ? (
                <p className="text-red-400 font-bold text-lg">Something went wrong. Please try again!</p>
              ) : (
                <>
                  <p className="text-xl font-semibold mb-2">
                    <span className="mr-2">Prediction:</span>
                    <span className={
                      result.prediction === "real"
                        ? "bg-gradient-to-r from-green-400 to-blue-400 text-transparent bg-clip-text animate-pulse"
                        : "bg-gradient-to-r from-red-400 to-pink-400 text-transparent bg-clip-text animate-pulse"
                    }>
                      {result.prediction.toUpperCase()}
                    </span>
                  </p>
                  <p className="text-lg">
                    <span className="mr-2">Confidence:</span>
                    <span className="font-mono text-yellow-300">
                      {(result.confidence).toFixed(2)}%
                    </span>
                  </p>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </main>
  );
}

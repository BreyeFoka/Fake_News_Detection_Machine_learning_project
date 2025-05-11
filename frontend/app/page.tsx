"use client";
import { useState } from "react";

export default function Home() {
  const [headline, setHeadline] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState<null | { prediction: string; confidence: number }>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headline, text }),
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="min-h-screen p-6 bg-gray-50 text-gray-800">
      <h1 className="text-2xl font-bold mb-4">🧠 Fake News Detector</h1>
      <form onSubmit={handleSubmit} className="space-y-4 max-w-xl">
        <input
          type="text"
          value={headline}
          onChange={(e) => setHeadline(e.target.value)}
          placeholder="Headline"
          className="w-full border p-2 rounded"
          required
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="News Text"
          rows={6}
          className="w-full border p-2 rounded"
          required
        />
        <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          Predict
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 border rounded bg-white">
          <p>
            <strong>Prediction:</strong>{" "}
            <span className={result.prediction === "real" ? "text-green-600" : "text-red-600"}>
              {result.prediction.toUpperCase()}
            </span>
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

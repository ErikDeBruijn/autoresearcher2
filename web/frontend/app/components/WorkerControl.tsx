"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function WorkerControl({ onClose }: { onClose: () => void }) {
  const [running, setRunning] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [cudaDevice, setCudaDevice] = useState("1");
  const [maxCycles, setMaxCycles] = useState(50);

  const checkStatus = () => {
    fetch(`${API}/api/workers/status`)
      .then((r) => r.json())
      .then((d) => setRunning(d.running))
      .catch(() => setRunning(null));
  };

  useEffect(() => {
    checkStatus();
    const id = setInterval(checkStatus, 5000);
    return () => clearInterval(id);
  }, []);

  const start = async () => {
    setLoading(true);
    await fetch(`${API}/api/workers/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cuda_device: cudaDevice, max_cycles: maxCycles }),
    });
    setLoading(false);
    checkStatus();
  };

  const stop = async () => {
    setLoading(true);
    await fetch(`${API}/api/workers/stop`, { method: "POST" });
    setLoading(false);
    setTimeout(checkStatus, 1000);
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold mb-4">Worker Control</h2>

        <div className="flex items-center gap-3 mb-4">
          <span
            className={`w-3 h-3 rounded-full ${
              running === null
                ? "bg-gray-600"
                : running
                ? "bg-green-400 animate-pulse"
                : "bg-red-400"
            }`}
          />
          <span className="text-sm">
            {running === null
              ? "Checking..."
              : running
              ? "Research loop running"
              : "Research loop stopped"}
          </span>
        </div>

        {!running && (
          <div className="space-y-3 mb-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                CUDA Device
              </label>
              <select
                value={cudaDevice}
                onChange={(e) => setCudaDevice(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              >
                <option value="0">GPU 0</option>
                <option value="1">GPU 1</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Max Cycles
              </label>
              <input
                type="number"
                value={maxCycles}
                onChange={(e) => setMaxCycles(Number(e.target.value))}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              />
            </div>
          </div>
        )}

        <div className="flex gap-2">
          {running ? (
            <button
              onClick={stop}
              disabled={loading}
              className="flex-1 px-4 py-2 text-sm bg-red-600 hover:bg-red-500 rounded font-medium disabled:opacity-50"
            >
              {loading ? "Stopping..." : "Stop Worker"}
            </button>
          ) : (
            <button
              onClick={start}
              disabled={loading}
              className="flex-1 px-4 py-2 text-sm bg-green-600 hover:bg-green-500 rounded font-medium disabled:opacity-50"
            >
              {loading ? "Starting..." : "Start Worker"}
            </button>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

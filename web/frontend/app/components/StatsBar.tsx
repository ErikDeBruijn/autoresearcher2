"use client";

import { useEffect, useState } from "react";

interface Stats {
  total_observations: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
  total_wall_time_s: number;
  world_model_version: number;
  belief_count: number;
  tension_count: number;
  queue_counts: Record<string, number>;
  workers: Record<string, { experiments: number; avg_time_s: number; total_time_s: number }>;
  total_energy_kwh: number;
  total_cost_eur: number;
}

interface GpuInfo {
  utilization_pct: number;
  power_w: number;
  temperature_c?: number | null;
  vram_used_gb?: number | null;
  vram_total_gb?: number | null;
}

interface EnergyStatus {
  shelly_total_w: number | null;
  system_base_w: number | null;
  price_eur_per_kwh: number | null;
}

interface WorkerStatus {
  running: boolean;
  gpus: GpuInfo[] | null;
  energy: EnergyStatus | null;
}

const API = process.env.NEXT_PUBLIC_API_URL || "";

export default function StatsBar() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus | null>(null);


  useEffect(() => {
    const fetchStats = () =>
      fetch(`${API}/api/stats`)
        .then((r) => r.json())
        .then(setStats)
        .catch(() => {});
    const fetchWorkers = () =>
      fetch(`${API}/api/workers/status`)
        .then((r) => r.json())
        .then(setWorkerStatus)
        .catch(() => {});
    fetchStats();
    fetchWorkers();
    const id = setInterval(() => { fetchStats(); fetchWorkers(); }, 10000);
    return () => clearInterval(id);
  }, []);

  if (!stats) return <div className="h-12 bg-gray-900 border-b border-gray-800 animate-pulse" />;

  const totalQueue = Object.values(stats.queue_counts).reduce((a, b) => a + b, 0);
  const workers = Object.entries(stats.workers);
  const anyIdle = workers.length === 0 || workers.every(([, w]) => w.experiments === 0);

  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-gray-900 border-b border-gray-800 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Experiments:</span>
        <span className="font-mono font-bold">{stats.total_observations}</span>
        <span className="text-green-400">({(stats.success_rate * 100).toFixed(0)}% ok)</span>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-gray-400">World Model:</span>
        <span className="font-mono">v{stats.world_model_version}</span>
        <span className="text-blue-400">{stats.belief_count} beliefs</span>
        {stats.tension_count > 0 && (
          <span className="text-yellow-400">{stats.tension_count} tensions</span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <span className="text-gray-400">Queue:</span>
        <span className="font-mono">{totalQueue}</span>
        <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
          {["backlog", "todo", "running", "done", "reviewed"].map((stage) => {
            const pct = totalQueue > 0 ? (stats.queue_counts[stage] / totalQueue) * 100 : 0;
            const colors: Record<string, string> = {
              backlog: "bg-gray-500",
              todo: "bg-blue-500",
              running: "bg-yellow-500",
              done: "bg-green-500",
              reviewed: "bg-purple-500",
            };
            return <div key={stage} className={`h-full ${colors[stage]}`} style={{ width: `${pct}%` }} />;
          })}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-gray-400">GPU Time:</span>
        <span className="font-mono">{(stats.total_wall_time_s / 60).toFixed(0)}min</span>
      </div>

      <div className="flex items-center gap-2">
        <a href="http://pve03.home:8377/" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-emerald-400 transition-colors">Energy:</a>
        <span className="font-mono text-emerald-400">{stats.total_cost_eur.toFixed(2)}€</span>
        <span className="font-mono text-gray-500">{(stats.total_energy_kwh * 1000).toFixed(0)}Wh</span>
      </div>

      <a
        href={`${API}/api/report/download`}
        download
        target="_blank"
        rel="noopener noreferrer"
        className="px-2 py-0.5 text-xs font-mono rounded border border-gray-600 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors no-underline"
        title="Generate and download PDF report"
      >
        PDF
      </a>

      <div className="flex items-center gap-3 ml-auto">
        {workerStatus?.energy?.price_eur_per_kwh != null && (
          <a href="http://pve03.home:8377/" target="_blank" rel="noopener noreferrer" className="text-xs text-gray-500 font-mono hover:text-emerald-400 transition-colors">{workerStatus.energy.price_eur_per_kwh.toFixed(3)}€/kWh</a>
        )}
        {workerStatus?.energy?.shelly_total_w != null && (
          <a href="http://pve03.home:8377/" target="_blank" rel="noopener noreferrer" className="text-xs text-gray-400 font-mono hover:text-emerald-400 transition-colors">{Math.round(workerStatus.energy.shelly_total_w)}W total</a>
        )}
        {workerStatus?.gpus ? (
          workerStatus.gpus.map((gpu, i) => {
            const tempColor = gpu.temperature_c != null
              ? gpu.temperature_c > 85 ? "text-red-400" : gpu.temperature_c > 75 ? "text-yellow-400" : "text-gray-400"
              : "";
            return (
              <div key={i} className="flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${gpu.utilization_pct > 10 ? "bg-green-400 animate-pulse" : "bg-gray-600"}`} />
                <span className="text-xs font-mono text-gray-300">GPU{i}: {gpu.utilization_pct}%</span>
                {gpu.vram_used_gb != null && gpu.vram_total_gb != null && (
                  <span className="text-xs font-mono text-gray-500">{gpu.vram_used_gb.toFixed(1)}/{gpu.vram_total_gb.toFixed(0)}GB</span>
                )}
                {gpu.temperature_c != null && (
                  <span className={`text-xs font-mono ${tempColor}`}>{gpu.temperature_c}°C</span>
                )}
                <span className="text-xs font-mono text-gray-600">{Math.round(gpu.power_w)}W</span>
              </div>
            );
          })
        ) : workers.length > 0 ? (
          workers.map(([wid, w]) => (
            <div key={wid} className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${w.experiments > 0 ? "bg-green-400 animate-pulse" : "bg-gray-600"}`} />
              <span className="text-xs text-gray-400">{wid.replace("worker_dllm-experiment_", "GPU")}</span>
            </div>
          ))
        ) : (
          <span className="text-xs text-red-400">No workers</span>
        )}
      </div>
    </div>
  );
}

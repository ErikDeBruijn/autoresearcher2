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
}

const API = process.env.NEXT_PUBLIC_API_URL || "";

export default function StatsBar() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    const fetch_ = () =>
      fetch(`${API}/api/stats`)
        .then((r) => r.json())
        .then(setStats)
        .catch(() => {});
    fetch_();
    const id = setInterval(fetch_, 10000);
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

      <div className="flex items-center gap-2 ml-auto">
        {workers.length > 0 ? (
          workers.map(([wid, w]) => (
            <div key={wid} className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${w.experiments > 0 ? "bg-green-400 animate-pulse" : "bg-gray-600"}`} />
              <span className="text-xs text-gray-400">{wid}</span>
              <span className="text-xs font-mono">{w.experiments}x</span>
            </div>
          ))
        ) : (
          <span className={`text-xs ${anyIdle ? "text-red-400" : "text-gray-500"}`}>
            {anyIdle ? "⚠ Workers idle" : "No workers"}
          </span>
        )}
      </div>
    </div>
  );
}

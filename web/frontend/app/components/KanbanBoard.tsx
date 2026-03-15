"use client";

import { useEffect, useState } from "react";
import ProposalCard from "./ProposalCard";
import type { Proposal } from "./ProposalCard";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const STAGES = [
  { key: "backlog", label: "Backlog", subtitle: "Generator output", color: "border-gray-600", dot: "bg-gray-400" },
  { key: "todo", label: "Todo", subtitle: "Critic approved", color: "border-blue-600", dot: "bg-blue-400" },
  { key: "running", label: "Running", subtitle: "Worker executing", color: "border-yellow-600", dot: "bg-yellow-400" },
  { key: "done", label: "Done", subtitle: "Awaiting review", color: "border-green-600", dot: "bg-green-400" },
  { key: "reviewed", label: "Reviewed", subtitle: "Observation analyzed", color: "border-purple-600", dot: "bg-purple-400" },
] as const;

export default function KanbanBoard() {
  const [queue, setQueue] = useState<Record<string, Proposal[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchQueue = () =>
      fetch(`${API}/api/queue`)
        .then((r) => r.json())
        .then((data) => {
          setQueue(data);
          setLoading(false);
        })
        .catch(() => setLoading(false));

    fetchQueue();
    const id = setInterval(fetchQueue, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading) {
    return (
      <div className="flex gap-4 p-4 h-full">
        {STAGES.map((s) => (
          <div key={s.key} className="flex-1 bg-gray-900/50 rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="flex gap-3 p-4 h-full overflow-x-auto">
      {STAGES.map((stage) => {
        const proposals = queue[stage.key] || [];
        return (
          <div
            key={stage.key}
            className={`flex-1 min-w-[240px] flex flex-col rounded-xl bg-gray-900/30 border-t-2 ${stage.color}`}
          >
            <div className="px-3 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${stage.dot}`} />
                <h3 className="font-semibold text-sm">{stage.label}</h3>
              </div>
              <span className="text-xs font-mono bg-gray-800 px-2 py-0.5 rounded-full text-gray-400">
                {proposals.length}
              </span>
            </div>
            <p className="px-3 text-xs text-gray-500 mb-2">{stage.subtitle}</p>
            <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-2">
              {proposals.map((p) => (
                <ProposalCard key={p.id} proposal={p} />
              ))}
              {proposals.length === 0 && (
                <div className="text-xs text-gray-600 text-center py-4">No items</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

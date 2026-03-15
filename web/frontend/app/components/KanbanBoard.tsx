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

interface ProposalWithWorker extends Proposal {
  worker_id?: string;
}

interface WorkerInfo {
  experiments: number;
  avg_time_s: number;
  total_time_s: number;
}

function WorkerSlot({
  workerId,
  proposals,
  info,
}: {
  workerId: string;
  proposals: ProposalWithWorker[];
  info?: WorkerInfo;
}) {
  const active = proposals.length > 0;
  return (
    <div className={`rounded-lg border ${active ? "border-yellow-700 bg-yellow-950/20" : "border-gray-700 bg-gray-800/20"} p-2`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`w-2 h-2 rounded-full ${active ? "bg-yellow-400 animate-pulse" : "bg-gray-600"}`} />
        <span className="text-xs font-medium">{workerId}</span>
        {info && (
          <span className="text-xs text-gray-500 ml-auto">{info.experiments}x runs</span>
        )}
      </div>
      {proposals.length > 0 ? (
        <div className="space-y-2">
          {proposals.map((p) => (
            <ProposalCard key={p.id} proposal={p} />
          ))}
        </div>
      ) : (
        <div className="text-xs text-gray-600 text-center py-2">Idle</div>
      )}
    </div>
  );
}

export default function KanbanBoard() {
  const [queue, setQueue] = useState<Record<string, ProposalWithWorker[]>>({});
  const [workers, setWorkers] = useState<Record<string, WorkerInfo>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [queueRes, statsRes] = await Promise.all([
          fetch(`${API}/api/queue`).then((r) => r.json()),
          fetch(`${API}/api/stats`).then((r) => r.json()),
        ]);
        setQueue(queueRes);
        setWorkers(statsRes.workers || {});
        setLoading(false);
      } catch {
        setLoading(false);
      }
    };

    fetchData();
    const id = setInterval(fetchData, 5000);
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

  // Build worker slots for the Running column
  const runningProposals = (queue["running"] || []) as ProposalWithWorker[];

  // Determine worker IDs: known workers from stats + any from running proposals
  const workerIds = new Set<string>();
  Object.keys(workers).forEach((w) => workerIds.add(w));
  runningProposals.forEach((p) => {
    if (p.worker_id) workerIds.add(p.worker_id);
  });

  const sortedWorkerIds = Array.from(workerIds).sort();
  const proposalsByWorker: Record<string, ProposalWithWorker[]> = {};
  for (const wid of sortedWorkerIds) {
    proposalsByWorker[wid] = runningProposals.filter((p) => p.worker_id === wid);
  }
  // Unassigned running proposals
  const unassigned = runningProposals.filter(
    (p) => !p.worker_id || !workerIds.has(p.worker_id)
  );

  return (
    <div className="flex gap-3 p-4 h-full overflow-x-auto">
      {STAGES.map((stage) => {
        const proposals = queue[stage.key] || [];

        // Running column gets special worker-slot treatment
        if (stage.key === "running") {
          return (
            <div
              key={stage.key}
              className={`flex-1 min-w-[280px] flex flex-col rounded-xl bg-gray-900/30 border-t-2 ${stage.color}`}
            >
              <div className="px-3 py-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${stage.dot}`} />
                  <h3 className="font-semibold text-sm">{stage.label}</h3>
                </div>
                <span className="text-xs font-mono bg-gray-800 px-2 py-0.5 rounded-full text-gray-400">
                  {runningProposals.length}
                </span>
              </div>
              <p className="px-3 text-xs text-gray-500 mb-2">{stage.subtitle}</p>
              <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-2">
                {sortedWorkerIds.map((wid) => (
                  <WorkerSlot
                    key={wid}
                    workerId={wid}
                    proposals={proposalsByWorker[wid] || []}
                    info={workers[wid]}
                  />
                ))}
                {unassigned.length > 0 && (
                  <div className="border border-gray-700 rounded-lg p-2">
                    <div className="text-xs text-gray-500 mb-2">Unassigned</div>
                    {unassigned.map((p) => (
                      <ProposalCard key={p.id} proposal={p} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        }

        // Regular columns
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

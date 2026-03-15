"use client";

import { useEffect, useState, useCallback } from "react";
import ProposalCard from "./ProposalCard";
import type { Proposal } from "./ProposalCard";

const API = process.env.NEXT_PUBLIC_API_URL || "";

const STAGES = [
  { key: "backlog", label: "Backlog", subtitle: "Generator output", color: "border-gray-600", dot: "bg-gray-400" },
  { key: "todo", label: "Todo", subtitle: "Critic approved", color: "border-blue-600", dot: "bg-blue-400" },
  { key: "running", label: "Running", subtitle: "Worker executing", color: "border-yellow-600", dot: "bg-yellow-400" },
  { key: "done", label: "Done", subtitle: "Awaiting review", color: "border-green-600", dot: "bg-green-400" },
  { key: "reviewed", label: "Reviewed", subtitle: "Observation analyzed", color: "border-purple-600", dot: "bg-purple-400" },
] as const;

const DROP_TARGETS = new Set(["backlog", "todo"]);

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
    <div className={`rounded-lg border ${active ? "border-yellow-700 bg-yellow-950" : "border-gray-700 bg-gray-800"} p-2`}>
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
  const [dragging, setDragging] = useState(false);
  const [dragOverStage, setDragOverStage] = useState<string | null>(null);
  const [trashOver, setTrashOver] = useState(false);

  const fetchData = useCallback(async () => {
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
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 5000);
    return () => clearInterval(id);
  }, [fetchData]);

  const moveProposal = async (proposalId: string, targetStage: string) => {
    await fetch(`${API}/api/proposals/${proposalId}/promote?target_stage=${targetStage}`, {
      method: "POST",
    });
    fetchData();
  };

  const deleteProposal = async (proposalId: string) => {
    await fetch(`${API}/api/proposals/${proposalId}`, { method: "DELETE" });
    fetchData();
  };

  const handleDragStart = (e: React.DragEvent, proposalId: string) => {
    e.dataTransfer.setData("text/plain", proposalId);
    e.dataTransfer.effectAllowed = "move";
    setDragging(true);
  };

  const handleDragEnd = () => {
    setDragging(false);
    setDragOverStage(null);
    setTrashOver(false);
  };

  const handleDragOver = (e: React.DragEvent, stageKey: string) => {
    if (DROP_TARGETS.has(stageKey)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      setDragOverStage(stageKey);
    }
  };

  const handleDragLeave = (e: React.DragEvent, stageKey: string) => {
    if (dragOverStage === stageKey) {
      setDragOverStage(null);
    }
  };

  const handleDrop = (e: React.DragEvent, stageKey: string) => {
    e.preventDefault();
    const proposalId = e.dataTransfer.getData("text/plain");
    if (proposalId && DROP_TARGETS.has(stageKey)) {
      moveProposal(proposalId, stageKey);
    }
    setDragOverStage(null);
    setDragging(false);
  };

  const handleTrashDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const proposalId = e.dataTransfer.getData("text/plain");
    if (proposalId) {
      deleteProposal(proposalId);
    }
    setTrashOver(false);
    setDragging(false);
  };

  if (loading) {
    return (
      <div className="flex gap-4 p-4 h-full">
        {STAGES.map((s) => (
          <div key={s.key} className="flex-1 bg-gray-900 rounded-xl animate-pulse" />
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

  const isDraggable = (stageKey: string) => stageKey === "backlog" || stageKey === "todo";
  const isDropTarget = (stageKey: string) => DROP_TARGETS.has(stageKey);

  return (
    <div className="flex flex-col h-full relative">
      <div className="flex gap-3 p-4 flex-1 min-h-0 overflow-x-auto">
        {STAGES.map((stage) => {
          const proposals = queue[stage.key] || [];
          const dropHighlight = dragOverStage === stage.key;

          // Running column gets special worker-slot treatment
          if (stage.key === "running") {
            return (
              <div
                key={stage.key}
                className={`flex-1 min-w-[280px] flex flex-col rounded-xl bg-gray-900 border-t-2 ${stage.color}`}
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
              className={`flex-1 min-w-[240px] flex flex-col rounded-xl bg-gray-900 border-t-2 transition-colors ${stage.color} ${
                dropHighlight ? "ring-2 ring-blue-500 bg-gray-800" : ""
              }`}
              onDragOver={(e) => handleDragOver(e, stage.key)}
              onDragLeave={(e) => handleDragLeave(e, stage.key)}
              onDrop={(e) => handleDrop(e, stage.key)}
              data-drop-stage={isDropTarget(stage.key) ? stage.key : undefined}
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
                  <div
                    key={p.id}
                    draggable={isDraggable(stage.key)}
                    onDragStart={(e) => handleDragStart(e, p.id)}
                    onDragEnd={handleDragEnd}
                    className={isDraggable(stage.key) ? "cursor-grab active:cursor-grabbing" : ""}
                    data-proposal-id={p.id}
                  >
                    <ProposalCard proposal={p} />
                  </div>
                ))}
                {proposals.length === 0 && !dropHighlight && (
                  <div className="text-xs text-gray-600 text-center py-4">No items</div>
                )}
                {dropHighlight && (
                  <div className="border-2 border-dashed border-blue-500 rounded-lg p-4 text-center text-xs text-blue-400">
                    Drop here
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Trash zone — visible when dragging, collapsed otherwise */}
      <div
        className={`mx-4 mb-4 rounded-xl border-2 border-dashed transition-all text-center ${
          dragging
            ? trashOver
              ? "border-red-500 bg-red-950 text-red-300 py-3"
              : "border-gray-600 bg-gray-900 text-gray-500 py-3"
            : "border-transparent py-0 h-0 overflow-hidden opacity-0"
        }`}
        onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; setTrashOver(true); }}
        onDragLeave={() => setTrashOver(false)}
        onDrop={handleTrashDrop}
        data-drop-stage="trash"
      >
        <span className="text-sm">{trashOver ? "Release to delete" : "Drag here to delete"}</span>
      </div>
    </div>
  );
}

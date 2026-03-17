"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import ProposalCard from "./ProposalCard";
import type { Proposal } from "./ProposalCard";
import ProjectFilter, { getProjectColor } from "./ProjectFilter";
import type { Project } from "./ProjectFilter";

const API = process.env.NEXT_PUBLIC_API_URL || "";

const STAGES = [
  { key: "backlog", label: "Proposed", subtitle: "Ready for critic", color: "border-gray-600", dot: "bg-gray-400" },
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
  selectedObservationId,
  onSelectObservation,
  onDragStart,
  onDragEnd,
}: {
  workerId: string;
  proposals: ProposalWithWorker[];
  info?: WorkerInfo;
  selectedObservationId?: string;
  onSelectObservation?: (obsId: string) => void;
  onDragStart?: (e: React.DragEvent, proposalId: string) => void;
  onDragEnd?: () => void;
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
            <div
              key={p.id}
              draggable
              onDragStart={(e) => onDragStart?.(e, p.id)}
              onDragEnd={onDragEnd}
              className="cursor-grab active:cursor-grabbing"
            >
              <ProposalCard proposal={p} isHighlighted={!!selectedObservationId && p.observation_id === selectedObservationId} onSelectObservation={onSelectObservation} />
            </div>
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
  const [projects, setProjects] = useState<Project[]>([]);
  const [visibleProjects, setVisibleProjects] = useState<Set<string>>(new Set());
  const [selectedObservationId, setSelectedObservationId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [dragOverStage, setDragOverStage] = useState<string | null>(null);
  const [trashOver, setTrashOver] = useState(false);
  const [pipelineActivity, setPipelineActivity] = useState<{ phase?: string; proposal_id?: string } | null>(null);
  const [activeWorkerIds, setActiveWorkerIds] = useState<Set<string>>(new Set());

  const toggleProject = useCallback((projectId: string) => {
    setVisibleProjects((prev) => {
      const next = new Set(prev);
      if (next.has(projectId)) {
        next.delete(projectId);
      } else {
        next.add(projectId);
      }
      return next;
    });
  }, []);

  const toggleProjectActive = useCallback(async (projectId: string, active: boolean) => {
    // Optimistic update
    setProjects((prev) => prev.map((p) => p.id === projectId ? { ...p, active } : p));
    try {
      await fetch(`${API}/api/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active }),
      });
    } catch (e) {
      console.error("Failed to toggle project active state", e);
      // Revert on failure
      setProjects((prev) => prev.map((p) => p.id === projectId ? { ...p, active: !active } : p));
    }
  }, []);

  const setProjectPriority = useCallback(async (projectId: string, priority: string) => {
    // Optimistic update
    setProjects((prev) => prev.map((p) => p.id === projectId ? { ...p, priority } : p));
    try {
      await fetch(`${API}/api/projects/${projectId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ priority }),
      });
    } catch (e) {
      console.error("Failed to set project priority", e);
    }
  }, []);

  // Build project lookup for enriching proposals with name/color
  const projectMap = new Map(projects.map((p, i) => [p.id, { ...p, color: getProjectColor(i) }]));

  // Compute which proposals set a new record (per project, using target metric + optimize direction)
  const recordIds = useMemo(() => {
    const allProposals: ProposalWithWorker[] = [];
    for (const stage of Object.values(queue)) {
      if (Array.isArray(stage)) allProposals.push(...stage);
    }

    // Build per-project metric config
    const metricConfig: Record<string, { metric: string; maximize: boolean }> = {};
    for (const p of projects) {
      metricConfig[p.id] = {
        metric: p.domain_config?.target_metric || "target_metric",
        maximize: p.domain_config?.optimize === "maximize",
      };
    }

    const sorted = allProposals
      .filter((p) => p.observation?.outcome_success && p.observation?.outcome_metrics)
      .sort((a, b) => (a.created_at || 0) - (b.created_at || 0));

    const records = new Set<string>();
    const bestByProject: Record<string, number> = {};
    for (const p of sorted) {
      const pid = p.project_id || "__none__";
      const cfg = p.project_id ? metricConfig[p.project_id] : { metric: "target_metric", maximize: false };
      if (!cfg) continue;
      const val = p.observation!.outcome_metrics![cfg.metric];
      if (val == null) continue;
      if (bestByProject[pid] === undefined || (cfg.maximize ? val > bestByProject[pid] : val < bestByProject[pid])) {
        bestByProject[pid] = val;
        records.add(p.id);
      }
    }
    return records;
  }, [queue, projects]);

  const enrichProposal = useCallback((p: ProposalWithWorker): ProposalWithWorker => {
    const proj = p.project_id ? projectMap.get(p.project_id) : null;
    return {
      ...p,
      project_name: proj?.name,
      project_color: proj?.color,
      is_record: recordIds.has(p.id),
      target_metric: proj?.domain_config?.target_metric || "target_metric",
      optimize: proj?.domain_config?.optimize || "minimize",
    };
  }, [projectMap, recordIds]);

  const isVisible = useCallback((p: ProposalWithWorker): boolean => {
    if (visibleProjects.size === 0) return true; // No filter = show all
    if (!p.project_id) return visibleProjects.has("__none__");
    return visibleProjects.has(p.project_id);
  }, [visibleProjects]);

  const fetchData = useCallback(async () => {
    try {
      const [queueRes, statsRes, projectsRes] = await Promise.all([
        fetch(`${API}/api/queue`).then((r) => r.json()),
        fetch(`${API}/api/stats`).then((r) => r.json()),
        fetch(`${API}/api/projects`).then((r) => r.json()).catch(() => []),
      ]);
      setQueue(queueRes);
      setWorkers(statsRes.workers || {});
      setActiveWorkerIds(new Set(statsRes.active_worker_ids || []));
      setPipelineActivity(statsRes.pipeline_activity || null);
      setProjects(projectsRes);
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

  // Update page title with active projects and their best scores
  useEffect(() => {
    const getEmoji = (name: string) => {
      const lower = name.toLowerCase();
      if (lower.includes("atari") || lower.includes("breakout") || lower.includes("pong")) return "🕹️";
      if (lower.includes("gpt") || lower.includes("llm") || lower.includes("nano")) return "💬";
      return "🔬";
    };

    const activeProjects = projects.filter((p) => p.active);
    if (activeProjects.length === 0) {
      document.title = "AR2";
      return;
    }
    // Compute best score per active project from queue data
    const allProposals: ProposalWithWorker[] = [];
    for (const stage of Object.values(queue)) {
      if (Array.isArray(stage)) allProposals.push(...stage);
    }
    const parts = activeProjects.map((proj) => {
      const metric = proj.domain_config?.target_metric || "target_metric";
      const maximize = proj.domain_config?.optimize === "maximize";
      let best: number | null = null;
      for (const p of allProposals) {
        if (p.project_id !== proj.id) continue;
        const val = p.observation?.outcome_metrics?.[metric];
        if (val != null && p.observation?.outcome_success) {
          if (best === null || (maximize ? val > best : val < best)) best = val;
        }
      }
      const emoji = getEmoji(proj.name);
      const score = best != null ? best.toFixed(4) : "";
      return `${emoji}${score}`;
    });
    document.title = `AR2 ${parts.join(" ")}`;
  }, [projects, queue]);

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

  const handleDragStart = (e: React.DragEvent, proposalId: string, fromStage?: string) => {
    e.dataTransfer.setData("text/plain", proposalId);
    e.dataTransfer.setData("application/x-stage", fromStage || "");
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

  const cancelProposal = async (proposalId: string) => {
    await fetch(`${API}/api/proposals/${proposalId}/cancel`, { method: "POST" });
    fetchData();
  };

  const handleDrop = (e: React.DragEvent, stageKey: string) => {
    e.preventDefault();
    const proposalId = e.dataTransfer.getData("text/plain");
    const fromStage = e.dataTransfer.getData("application/x-stage");
    if (proposalId && DROP_TARGETS.has(stageKey)) {
      if (fromStage === "running") {
        if (window.confirm("Cancel this running experiment and move it back to Proposed?")) {
          cancelProposal(proposalId);
        }
      } else {
        moveProposal(proposalId, stageKey);
      }
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

  // Build worker slots for the Running column (always unfiltered)
  const runningProposals = ((queue["running"] || []) as ProposalWithWorker[]).map(enrichProposal);

  // Determine worker IDs: active workers + any from running proposals
  // (excludes old/retired workers that only have historical data)
  const workerIds = new Set<string>();
  activeWorkerIds.forEach((w) => workerIds.add(w));
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

  const isDraggable = (stageKey: string) => stageKey === "backlog" || stageKey === "todo" || stageKey === "running";
  const isDropTarget = (stageKey: string) => DROP_TARGETS.has(stageKey);

  return (
    <div className="flex flex-col h-full relative">
      <ProjectFilter
        projects={projects}
        visibleProjects={visibleProjects}
        onToggle={toggleProject}
        onToggleActive={toggleProjectActive}
        onSetPriority={setProjectPriority}
        selectedObservationId={selectedObservationId}
        onSelectObservation={setSelectedObservationId}
      />
      <div className="flex gap-3 p-4 flex-1 min-h-0 overflow-x-auto">
        {STAGES.map((stage) => {
          const rawProposals = queue[stage.key] || [];
          // Running column always shows all projects (never filtered)
          const proposals = (stage.key === "running"
            ? rawProposals.map(enrichProposal)
            : rawProposals.map(enrichProposal).filter(isVisible)
          ).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
          const dropHighlight = dragOverStage === stage.key;

          // Pipeline activity: show which column has an active LLM process
          const phase = pipelineActivity?.phase;
          const stageActive =
            (stage.key === "done" && phase === "orienting") ||
            (stage.key === "backlog" && (phase === "critiquing" || phase === "generating"));

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
                      selectedObservationId={selectedObservationId}
                      onSelectObservation={setSelectedObservationId}
                      onDragStart={(e, id) => handleDragStart(e, id, "running")}
                      onDragEnd={handleDragEnd}
                    />
                  ))}
                  {unassigned.length > 0 && (
                    <div className="border border-gray-700 rounded-lg p-2">
                      <div className="text-xs text-gray-500 mb-2">Unassigned</div>
                      {unassigned.map((p) => (
                        <ProposalCard key={p.id} proposal={p} isHighlighted={!!selectedObservationId && p.observation_id === selectedObservationId} onSelectObservation={setSelectedObservationId} />
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
                  <span className={`w-2 h-2 rounded-full ${stageActive ? "bg-yellow-400 animate-pulse" : stage.dot}`} />
                  <h3 className="font-semibold text-sm">{stage.label}</h3>
                  {stageActive && (
                    <span className="text-xs text-yellow-400 animate-pulse">
                      {phase === "orienting" ? "reviewing..." : phase === "critiquing" ? "evaluating..." : "generating..."}
                    </span>
                  )}
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
                    <ProposalCard proposal={p} isHighlighted={!!selectedObservationId && p.observation_id === selectedObservationId} onSelectObservation={setSelectedObservationId} />
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

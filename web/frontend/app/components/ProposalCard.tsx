"use client";

import { useState, useEffect, useRef } from "react";

interface ObservationData {
  outcome_success: boolean;
  outcome_metrics: Record<string, number> | null;
  wall_time_s: number | null;
  error: string | null;
  energy_kwh: number | null;
  cost_eur: number | null;
  avg_power_w: number | null;
  artifact_paths: Record<string, string> | null;
}

interface WorldModelUpdate {
  version: number;
  reasoning: string | null;
  delta: Record<string, unknown>;
}

interface Proposal {
  id: string;
  created_at: number;
  status: string;
  intent: string;
  rationale: string;
  expected_learning: string;
  intervention_type: string;
  intervention_spec: Record<string, string>;
  estimated_cost: Record<string, string> | null;
  critic: { decision: string; rank: number; rationale: string } | null;
  observation_id: string | null;
  observation?: ObservationData;
  world_model_update?: WorldModelUpdate;
  project_id?: string | null;
  project_name?: string;
  project_color?: string;
  is_record?: boolean;
  target_metric?: string;
  optimize?: string;
  started_at?: number | null;
  promoted_at?: number | null;
  finished_at?: number | null;
}

const typeColors: Record<string, string> = {
  config_change: "bg-blue-900/50 text-blue-300 border-blue-700",
  probe: "bg-cyan-900/50 text-cyan-300 border-cyan-700",
  code_change: "bg-orange-900/50 text-orange-300 border-orange-700",
  replication: "bg-green-900/50 text-green-300 border-green-700",
};

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000 - ts);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function ProposalCard({
  proposal,
  isHighlighted,
  onSelectObservation,
}: {
  proposal: Proposal;
  isHighlighted?: boolean;
  onSelectObservation?: (obsId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isHighlighted) {
      setExpanded(true);
      cardRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [isHighlighted]);

  const handleClick = () => {
    setExpanded(!expanded);
    if (proposal.observation_id && onSelectObservation) {
      onSelectObservation(expanded ? "" : proposal.observation_id);
    }
  };

  return (
    <div
      ref={cardRef}
      className={`p-3 rounded-lg border cursor-pointer transition-all hover:border-gray-500 ${
        isHighlighted
          ? "ring-2 ring-cyan-500 border-cyan-600 bg-cyan-950/30"
          : proposal.status === "running" ? "border-yellow-600 bg-yellow-950 animate-pulse" : "border-gray-700 bg-gray-800"
      }`}
      onClick={handleClick}
    >
      <div className="flex items-start justify-between gap-2">
        <h4 className="text-sm font-medium leading-tight">{proposal.intent}</h4>
        {proposal.critic?.rank && (
          <span className="shrink-0 text-xs font-mono bg-gray-700 px-1.5 py-0.5 rounded">
            #{proposal.critic.rank}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 mt-2 flex-wrap">
        {proposal.project_name && (
          <span className="flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">
            <span className={`w-1.5 h-1.5 rounded-full ${proposal.project_color || "bg-gray-400"}`} />
            {proposal.project_name}
          </span>
        )}
        <span className={`text-xs px-1.5 py-0.5 rounded border ${typeColors[proposal.intervention_type] || "bg-gray-800 text-gray-400 border-gray-600"}`}>
          {proposal.intervention_type}
        </span>
        <span className="text-xs text-gray-500">{timeAgo(
          proposal.status === "running" ? (proposal.started_at || proposal.created_at)
          : proposal.status === "done" || proposal.status === "reviewed" ? (proposal.finished_at || proposal.created_at)
          : (proposal.promoted_at || proposal.created_at)
        )}</span>
        {proposal.is_record && (
          <span className="text-xs text-amber-400">🥇 New best</span>
        )}
        {proposal.observation?.artifact_paths?.video && (
          <span className="text-xs text-gray-500" title="Has gameplay video">🎬</span>
        )}
        {proposal.observation?.outcome_metrics && (() => {
          const metric = proposal.target_metric || "target_metric";
          const val = proposal.observation!.outcome_metrics![metric];
          if (val == null) return null;
          return (
            <span className="text-xs font-mono font-bold text-cyan-300">
              {metric}: {typeof val === "number" ? val.toFixed(4) : String(val)}
            </span>
          );
        })()}
        {proposal.observation?.cost_eur != null && (
          <span className="text-xs text-emerald-400">{proposal.observation.cost_eur.toFixed(3)}€</span>
        )}
      </div>

      {proposal.status === "todo" && proposal.critic?.rationale && (
        <div className="mt-1.5 text-xs text-blue-300 leading-tight">{proposal.critic.rationale}</div>
      )}

      {expanded && (
        <div className="mt-3 space-y-2 text-xs text-gray-400">
          <div>
            <span className="text-gray-500">Rationale:</span> {proposal.rationale}
          </div>
          <div>
            <span className="text-gray-500">Expected learning:</span> {proposal.expected_learning}
          </div>
          {Object.keys(proposal.intervention_spec).length > 0 && (
            <div className="font-mono bg-gray-900 p-2 rounded">
              {Object.entries(proposal.intervention_spec).map(([k, v]) => (
                <div key={k} className="truncate">{k}: {typeof v === "string" ? v : JSON.stringify(v).slice(0, 200)}</div>
              ))}
            </div>
          )}
          {proposal.critic && (
            <div className="border-t border-gray-700 pt-2">
              <span className="text-gray-500">Critic:</span> {proposal.critic.rationale}
            </div>
          )}
          {proposal.observation && (
            <div className="border-t border-gray-700 pt-2">
              <div className="flex items-center gap-2 mb-1">
                <span className={`font-medium ${proposal.observation.outcome_success ? "text-green-400" : "text-red-400"}`}>
                  {proposal.observation.outcome_success ? "Success" : "Failed"}
                </span>
                {proposal.observation.wall_time_s && (
                  <span className="text-gray-500">{Math.round(proposal.observation.wall_time_s)}s</span>
                )}
                {proposal.observation.cost_eur != null && (
                  <span className="text-emerald-400">{proposal.observation.cost_eur.toFixed(3)}€</span>
                )}
                {proposal.observation.energy_kwh != null && (
                  <span className="text-gray-500">{(proposal.observation.energy_kwh * 1000).toFixed(0)}Wh</span>
                )}
                {proposal.observation.avg_power_w != null && (
                  <span className="text-gray-500">{Math.round(proposal.observation.avg_power_w)}W</span>
                )}
              </div>
              {proposal.observation.outcome_metrics && (
                <div className="font-mono bg-gray-900 p-2 rounded">
                  {Object.entries(proposal.observation.outcome_metrics).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-gray-500">{k}:</span>{" "}
                      <span className={k === proposal.target_metric ? "text-cyan-300 font-bold" : ""}>
                        {typeof v === "number" ? v.toFixed(4) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {proposal.observation.error && (
                <div className="text-red-400 bg-red-950 p-2 rounded mt-1 break-all">
                  {proposal.observation.error.slice(0, 200)}
                </div>
              )}
              {proposal.observation.artifact_paths?.video && proposal.observation_id && (
                <div className="mt-2">
                  <video
                    src={`${process.env.NEXT_PUBLIC_API_URL || ""}/api/artifacts/${proposal.observation_id}/video`}
                    controls
                    muted
                    loop
                    className="w-full max-w-[300px] rounded border border-gray-700"
                    preload="metadata"
                  />
                </div>
              )}
            </div>
          )}
          {proposal.world_model_update && (
            <div className="border-t border-gray-700 pt-2">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-purple-400 font-medium">WM v{proposal.world_model_update.version}</span>
                {proposal.world_model_update.delta?.learntropy != null && (
                  <span className="text-xs text-gray-500">
                    learntropy: {(proposal.world_model_update.delta.learntropy as number).toFixed(3)}
                  </span>
                )}
              </div>
              {proposal.world_model_update.reasoning && (
                <div className="text-gray-400">{proposal.world_model_update.reasoning}</div>
              )}
              {proposal.world_model_update.delta && (() => {
                const d = proposal.world_model_update!.delta;
                const added = (d.beliefs_added as Array<{claim: string; confidence: number}>) || [];
                const revised = (d.beliefs_revised as Array<{id: string; new_confidence?: number; reason?: string}>) || [];
                const tensions = (d.tensions_added as Array<{description?: string; nature?: string}>) || [];
                if (added.length === 0 && revised.length === 0 && tensions.length === 0) return null;
                return (
                  <div className="mt-1 space-y-1">
                    {added.map((b, i) => (
                      <div key={`a${i}`} className="text-green-400">
                        + [{b.confidence?.toFixed(2)}] {b.claim}
                      </div>
                    ))}
                    {revised.map((r, i) => (
                      <div key={`r${i}`} className="text-yellow-400">
                        ~ {r.id} → {r.new_confidence?.toFixed(2)} {r.reason && `(${r.reason})`}
                      </div>
                    ))}
                    {tensions.map((t, i) => (
                      <div key={`t${i}`} className="text-orange-400">
                        ⚡ {t.description || t.nature}
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export type { Proposal };

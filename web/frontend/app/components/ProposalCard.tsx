"use client";

import { useState } from "react";

interface ObservationData {
  outcome_success: boolean;
  outcome_metrics: Record<string, number> | null;
  wall_time_s: number | null;
  error: string | null;
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

export default function ProposalCard({ proposal }: { proposal: Proposal }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-all hover:border-gray-500 ${
        proposal.status === "running" ? "border-yellow-600 bg-yellow-950 animate-pulse" : "border-gray-700 bg-gray-800"
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between gap-2">
        <h4 className="text-sm font-medium leading-tight">{proposal.intent}</h4>
        {proposal.critic?.rank && (
          <span className="shrink-0 text-xs font-mono bg-gray-700 px-1.5 py-0.5 rounded">
            #{proposal.critic.rank}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 mt-2">
        <span className={`text-xs px-1.5 py-0.5 rounded border ${typeColors[proposal.intervention_type] || "bg-gray-800 text-gray-400 border-gray-600"}`}>
          {proposal.intervention_type}
        </span>
        <span className="text-xs text-gray-500">{timeAgo(proposal.created_at)}</span>
        {proposal.observation_id && (
          <span className="text-xs text-green-400">✓ observed</span>
        )}
      </div>

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
                <div key={k}>{k}: {v}</div>
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
              </div>
              {proposal.observation.outcome_metrics && (
                <div className="font-mono bg-gray-900 p-2 rounded">
                  {Object.entries(proposal.observation.outcome_metrics).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-gray-500">{k}:</span>{" "}
                      <span className={k === "val_bpb" ? "text-cyan-300 font-bold" : ""}>
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
            </div>
          )}
          {proposal.world_model_update && (
            <div className="border-t border-gray-700 pt-2">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-purple-400 font-medium">WM v{proposal.world_model_update.version}</span>
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

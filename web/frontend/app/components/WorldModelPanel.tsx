"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "";

interface Belief {
  id: string;
  claim: string;
  confidence: number;
  evidence_for: string[];
  evidence_against: string[];
}

interface Tension {
  id: string;
  nature?: string;
  belief_ids?: string[];
  salience?: number;
}

interface WorldModelData {
  version: number;
  beliefs: Belief[];
  tensions: Tension[];
  cost_beliefs: Record<string, Record<string, number>>;
}

export default function WorldModelPanel({ onClose, projectId, projectName }: { onClose: () => void; projectId?: string; projectName?: string }) {
  const [wm, setWm] = useState<WorldModelData | null>(null);

  useEffect(() => {
    const url = projectId
      ? `${API}/api/world-model?project_id=${encodeURIComponent(projectId)}`
      : `${API}/api/world-model`;
    fetch(url)
      .then((r) => r.json())
      .then(setWm)
      .catch(() => {});
  }, [projectId]);

  if (!wm) return null;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">World Model v{wm.version}{projectName ? ` — ${projectName}` : ""}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">✕</button>
        </div>

        <section className="mb-4">
          <h3 className="text-sm font-semibold text-blue-400 mb-2">Beliefs ({wm.beliefs.length})</h3>
          <div className="space-y-2">
            {wm.beliefs.map((b) => (
              <div key={b.id} className="bg-gray-800 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden shrink-0">
                    <div
                      className={`h-full rounded-full ${
                        b.confidence >= 0.7 ? "bg-green-500" : b.confidence >= 0.4 ? "bg-yellow-500" : "bg-red-500"
                      }`}
                      style={{ width: `${b.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-500 shrink-0">{(b.confidence * 100).toFixed(0)}%</span>
                  <span className="text-sm">{b.claim}</span>
                </div>
                <div className="flex gap-4 mt-1 text-xs text-gray-500">
                  <span className="text-green-600">+{b.evidence_for.length} for</span>
                  <span className="text-red-600">-{b.evidence_against.length} against</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {wm.tensions.length > 0 && (
          <section className="mb-4">
            <h3 className="text-sm font-semibold text-yellow-400 mb-2">Tensions ({wm.tensions.length})</h3>
            {wm.tensions.map((t) => (
              <div key={t.id} className="bg-gray-800 rounded-lg p-3 mb-2">
                <div className="text-sm">{t.nature || "Unresolved tension"}</div>
                {t.belief_ids && (
                  <div className="text-xs text-gray-500 mt-1">Between: {t.belief_ids.join(", ")}</div>
                )}
              </div>
            ))}
          </section>
        )}

        {Object.keys(wm.cost_beliefs).length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-cyan-400 mb-2">Cost Beliefs</h3>
            <div className="bg-gray-800 rounded-lg p-3 font-mono text-xs">
              {Object.entries(wm.cost_beliefs).map(([type, costs]) => (
                <div key={type} className="flex gap-4">
                  <span className="text-gray-400 w-32">{type}:</span>
                  {Object.entries(costs).map(([k, v]) => (
                    <span key={k}>{k}={v}</span>
                  ))}
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

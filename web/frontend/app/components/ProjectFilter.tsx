"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface Project {
  id: string;
  name: string;
  description: string;
  active: boolean;
  domain_config?: { target_metric?: string } | null;
}

interface Observation {
  id: string;
  created_at: number;
  project_id: string | null;
  outcome_metrics: Record<string, number> | null;
  outcome_success: boolean;
  intervention_spec: Record<string, string>;
}

const PROJECT_COLORS: Record<number, string> = {
  0: "bg-blue-500",
  1: "bg-green-500",
  2: "bg-purple-500",
  3: "bg-orange-500",
  4: "bg-pink-500",
  5: "bg-cyan-500",
  6: "bg-yellow-500",
  7: "bg-red-500",
};

const PROJECT_HEX: Record<number, string> = {
  0: "#3b82f6",
  1: "#22c55e",
  2: "#a855f7",
  3: "#f97316",
  4: "#ec4899",
  5: "#06b6d4",
  6: "#eab308",
  7: "#ef4444",
};

export function getProjectColor(index: number): string {
  return PROJECT_COLORS[index % Object.keys(PROJECT_COLORS).length];
}

function getProjectHex(index: number): string {
  return PROJECT_HEX[index % Object.keys(PROJECT_HEX).length];
}

const API = process.env.NEXT_PUBLIC_API_URL || "";

interface ProgressChartProps {
  observations: Observation[];
  metric: string;
  color: string;
  selectedObservationId?: string;
  onSelectObservation?: (obsId: string) => void;
}

function ProgressChart({ observations, metric, color, selectedObservationId, onSelectObservation }: ProgressChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    run: number;
    value: number;
    isRecord: boolean;
    spec: Record<string, string>;
  } | null>(null);

  const width = 600;
  const height = 200;
  const pad = { top: 20, right: 20, bottom: 30, left: 55 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Sort by created_at, extract metric values
  const sorted = [...observations]
    .filter((o) => o.outcome_success && o.outcome_metrics?.[metric] != null)
    .sort((a, b) => a.created_at - b.created_at);

  if (sorted.length === 0) {
    return <div className="text-xs text-gray-500 py-4 text-center">No {metric} data yet</div>;
  }

  const values = sorted.map((o) => o.outcome_metrics![metric]);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 0.01;
  const yMin = minVal - range * 0.05;
  const yMax = maxVal + range * 0.05;

  const xScale = (i: number) => pad.left + (i / Math.max(sorted.length - 1, 1)) * plotW;
  const yScale = (v: number) => pad.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // Compute records (cumulative minimum for minimize)
  let bestSoFar = Infinity;
  const records: { run: number; value: number }[] = [];
  const points = sorted.map((o, i) => {
    const v = values[i];
    const isRecord = v < bestSoFar;
    if (isRecord) {
      bestSoFar = v;
      records.push({ run: i, value: v });
    }
    return { x: xScale(i), y: yScale(v), value: v, isRecord, spec: o.intervention_spec, run: i, obsId: o.id };
  });

  // Build step path for records
  let stepPath = "";
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    const x = xScale(r.run);
    const y = yScale(r.value);
    if (i === 0) {
      stepPath += `M ${x} ${y}`;
    } else {
      // Horizontal line from previous point to this x, then vertical drop
      stepPath += ` H ${x} V ${y}`;
    }
    // Extend to the right edge after the last record
    if (i === records.length - 1) {
      stepPath += ` H ${xScale(sorted.length - 1)}`;
    }
  }

  // Y-axis ticks
  const nTicks = 5;
  const yTicks = Array.from({ length: nTicks }, (_, i) => yMin + ((yMax - yMin) * i) / (nTicks - 1));

  const findNearest = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * width;
    let nearest = points[0];
    let minDist = Infinity;
    for (const p of points) {
      const d = Math.abs(p.x - mx);
      if (d < minDist) { minDist = d; nearest = p; }
    }
    return minDist < 30 ? nearest : null;
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const nearest = findNearest(e);
    if (nearest) {
      setTooltip({ x: nearest.x, y: nearest.y, run: nearest.run + 1, value: nearest.value, isRecord: nearest.isRecord, spec: nearest.spec });
    } else {
      setTooltip(null);
    }
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const nearest = findNearest(e);
    if (nearest && onSelectObservation) {
      onSelectObservation(selectedObservationId === nearest.obsId ? "" : nearest.obsId);
    }
  };

  // Find selected point for highlight
  const selectedPoint = selectedObservationId ? points.find((p) => p.obsId === selectedObservationId) : null;

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full max-w-[600px] h-auto"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
        onClick={handleClick}
        style={{ cursor: "crosshair" }}
      >
        {/* Grid lines */}
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={pad.left} y1={yScale(v)} x2={width - pad.right} y2={yScale(v)} stroke="#374151" strokeWidth={0.5} />
            <text x={pad.left - 5} y={yScale(v) + 3} textAnchor="end" fill="#6b7280" fontSize={10}>
              {v.toFixed(3)}
            </text>
          </g>
        ))}

        {/* X-axis labels */}
        {sorted.map((_, i) => {
          if (sorted.length <= 20 || i % Math.ceil(sorted.length / 10) === 0 || i === sorted.length - 1) {
            return (
              <text key={i} x={xScale(i)} y={height - 5} textAnchor="middle" fill="#6b7280" fontSize={10}>
                {i + 1}
              </text>
            );
          }
          return null;
        })}
        <text x={pad.left + plotW / 2} y={height} textAnchor="middle" fill="#4b5563" fontSize={9}>
          run #
        </text>

        {/* Step line for records (green) */}
        {stepPath && <path d={stepPath} fill="none" stroke={color} strokeWidth={2} />}

        {/* Data points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={p.isRecord ? 4 : 3}
            fill={p.isRecord ? color : "#4b5563"}
            stroke={p.isRecord ? color : "#6b7280"}
            strokeWidth={1}
            opacity={p.isRecord ? 1 : 0.6}
          />
        ))}

        {/* Selected point highlight */}
        {selectedPoint && (
          <>
            <circle cx={selectedPoint.x} cy={selectedPoint.y} r={8} fill="none" stroke="#06b6d4" strokeWidth={2} />
            <line x1={selectedPoint.x} y1={pad.top} x2={selectedPoint.x} y2={pad.top + plotH} stroke="#06b6d4" strokeWidth={1} strokeDasharray="4 2" opacity={0.5} />
          </>
        )}

        {/* Tooltip crosshair */}
        {tooltip && (
          <>
            <line x1={tooltip.x} y1={pad.top} x2={tooltip.x} y2={pad.top + plotH} stroke="#9ca3af" strokeWidth={0.5} strokeDasharray="4 2" />
            <circle cx={tooltip.x} cy={tooltip.y} r={6} fill="none" stroke="white" strokeWidth={1.5} />
          </>
        )}
      </svg>

      {/* Tooltip overlay */}
      {tooltip && (
        <div
          className="absolute bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs pointer-events-none z-10 shadow-lg"
          style={{
            left: `${(tooltip.x / width) * 100}%`,
            top: `${(tooltip.y / height) * 100 - 15}%`,
            transform: "translate(-50%, -100%)",
          }}
        >
          <div className="font-mono">
            <span className="text-gray-400">run {tooltip.run}:</span>{" "}
            <span className={tooltip.isRecord ? "text-green-400 font-bold" : "text-gray-300"}>
              {tooltip.value.toFixed(4)}
            </span>
            {tooltip.isRecord && <span className="text-green-400 ml-1">★</span>}
          </div>
          {Object.keys(tooltip.spec).length > 0 && (
            <div className="text-gray-500 mt-0.5">
              {Object.entries(tooltip.spec).slice(0, 3).map(([k, v]) => (
                <span key={k} className="mr-2">{k}={v}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ProjectFilter({
  projects,
  visibleProjects,
  onToggle,
  selectedObservationId,
  onSelectObservation,
}: {
  projects: Project[];
  visibleProjects: Set<string>;
  onToggle: (projectId: string) => void;
  selectedObservationId?: string;
  onSelectObservation?: (obsId: string) => void;
}) {
  const [expandedProject, setExpandedProject] = useState<string | null>(null);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [projectBest, setProjectBest] = useState<Record<string, number>>({});

  // Fetch observations for chart data
  useEffect(() => {
    const fetchObs = () =>
      fetch(`${API}/api/observations`)
        .then((r) => r.json())
        .then((obs: Observation[]) => {
          setObservations(obs);
          // Compute best val_bpb per project
          const best: Record<string, number> = {};
          for (const o of obs) {
            const pid = o.project_id || "__none__";
            const val = o.outcome_metrics?.val_bpb;
            if (val != null && o.outcome_success) {
              if (best[pid] === undefined || val < best[pid]) {
                best[pid] = val;
              }
            }
          }
          setProjectBest(best);
        })
        .catch(() => {});
    fetchObs();
    const id = setInterval(fetchObs, 10000);
    return () => clearInterval(id);
  }, []);

  if (projects.length === 0) return null;

  const getTargetMetric = (p: Project) => p.domain_config?.target_metric || "val_bpb";

  return (
    <div className="bg-gray-900 border-b border-gray-800">
      {projects.map((p, i) => {
        const visible = visibleProjects.has(p.id);
        const color = getProjectColor(i);
        const hex = getProjectHex(i);
        const expanded = expandedProject === p.id;
        const metric = getTargetMetric(p);
        const best = projectBest[p.id];
        const projObs = observations.filter((o) => o.project_id === p.id);

        return (
          <div key={p.id}>
            <div
              className="flex items-center gap-3 px-4 py-2 cursor-pointer hover:bg-gray-800/50 transition-colors"
              onClick={() => setExpandedProject(expanded ? null : p.id)}
            >
              {/* Visibility toggle (eye icon) */}
              <button
                onClick={(e) => { e.stopPropagation(); onToggle(p.id); }}
                className="text-sm transition-opacity"
                title={visible ? "Hide from kanban" : "Show in kanban"}
              >
                {visible ? "👁" : "👁‍🗨"}
              </button>

              {/* Project name */}
              <span className={`flex items-center gap-1.5 text-xs font-medium ${visible ? "text-gray-200" : "text-gray-500"}`}>
                <span className={`w-2 h-2 rounded-full ${color}`} />
                {p.name}
              </span>

              {/* Target metric + best value */}
              <span className="text-xs text-gray-500">minimize</span>
              <span className="text-xs font-mono text-cyan-300">{metric}</span>
              {best != null && (
                <>
                  <span className="text-xs text-gray-500">(lowest</span>
                  <span className="text-xs font-mono font-bold text-green-400">{best.toFixed(4)}</span>
                  <span className="text-xs text-gray-500">)</span>
                </>
              )}
              {projObs.length > 0 && (
                <span className="text-xs text-gray-600">{projObs.filter((o) => o.outcome_success).length} runs</span>
              )}

              <span className="text-xs text-gray-600 ml-auto">{expanded ? "▼" : "▶"}</span>

              {!p.active && <span className="text-xs text-gray-600">(paused)</span>}
            </div>

            {/* Collapsible chart */}
            {expanded && (
              <div className="px-4 pb-3">
                <ProgressChart
                  observations={projObs}
                  metric={metric}
                  color={hex}
                  selectedObservationId={selectedObservationId}
                  onSelectObservation={onSelectObservation}
                />
              </div>
            )}
          </div>
        );
      })}

      {/* Unassigned toggle */}
      <div className="flex items-center gap-3 px-4 py-1 border-t border-gray-800">
        <button
          onClick={() => onToggle("__none__")}
          className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-all ${
            visibleProjects.has("__none__")
              ? "bg-gray-800 text-gray-200 border border-gray-600"
              : "bg-gray-900 text-gray-600 border border-gray-800"
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${visibleProjects.has("__none__") ? "bg-gray-400" : "bg-gray-700"}`} />
          No project
        </button>
      </div>
    </div>
  );
}

export type { Project };

"use client";

interface Project {
  id: string;
  name: string;
  description: string;
  active: boolean;
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

export function getProjectColor(index: number): string {
  return PROJECT_COLORS[index % Object.keys(PROJECT_COLORS).length];
}

export default function ProjectFilter({
  projects,
  visibleProjects,
  onToggle,
}: {
  projects: Project[];
  visibleProjects: Set<string>;
  onToggle: (projectId: string) => void;
}) {
  if (projects.length === 0) return null;

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-gray-900 border-b border-gray-800">
      <span className="text-xs text-gray-500 font-medium">Projects:</span>
      {projects.map((p, i) => {
        const visible = visibleProjects.has(p.id);
        const color = getProjectColor(i);
        return (
          <button
            key={p.id}
            onClick={() => onToggle(p.id)}
            className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs transition-all ${
              visible
                ? "bg-gray-800 text-gray-200 border border-gray-600"
                : "bg-gray-900 text-gray-600 border border-gray-800"
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${visible ? color : "bg-gray-700"}`} />
            {p.name}
            {!p.active && <span className="text-gray-600">(paused)</span>}
          </button>
        );
      })}
      {/* Unassigned proposals toggle */}
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
  );
}

export type { Project };

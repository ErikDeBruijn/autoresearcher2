"use client";

import { useState } from "react";
import StatsBar from "./components/StatsBar";
import KanbanBoard from "./components/KanbanBoard";
import ProposalForm from "./components/ProposalForm";
import WorldModelPanel from "./components/WorldModelPanel";

export default function Dashboard() {
  const [showForm, setShowForm] = useState(false);
  const [showWorldModel, setShowWorldModel] = useState(false);

  return (
    <div className="h-screen flex flex-col">
      <StatsBar />

      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800">
        <h1 className="text-lg font-bold">AutoResearcher2</h1>
        <span className="text-xs text-gray-500">Generator-Critic Research Pipeline</span>
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => setShowWorldModel(true)}
            className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 rounded border border-gray-700"
          >
            World Model
          </button>
          <button
            onClick={() => setShowForm(true)}
            className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 rounded font-medium"
          >
            + New Proposal
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <KanbanBoard />
      </div>

      {showForm && <ProposalForm onClose={() => setShowForm(false)} />}
      {showWorldModel && <WorldModelPanel onClose={() => setShowWorldModel(false)} />}
    </div>
  );
}

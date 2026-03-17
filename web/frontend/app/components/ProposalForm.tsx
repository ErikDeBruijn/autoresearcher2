"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "";

export default function ProposalForm({ onClose }: { onClose: () => void }) {
  const [intent, setIntent] = useState("");
  const [rationale, setRationale] = useState("");
  const [expectedLearning, setExpectedLearning] = useState("");
  const [interventionType, setInterventionType] = useState("config_change");
  const [specText, setSpecText] = useState("{}");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      let spec = {};
      try {
        spec = JSON.parse(specText);
      } catch {
        setError("Invalid JSON in intervention spec");
        setSubmitting(false);
        return;
      }

      const res = await fetch(`${API}/api/proposals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intent,
          rationale,
          expected_learning: expectedLearning,
          intervention_type: interventionType,
          intervention_spec: spec,
        }),
      });

      if (!res.ok) throw new Error(await res.text());
      onClose();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold mb-4">New Proposal</h2>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Epistemic Intent</label>
            <input
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              placeholder="What belief or tension does this address?"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Rationale</label>
            <textarea
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              rows={2}
              placeholder="Why is this valuable now?"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Expected Learning</label>
            <textarea
              value={expectedLearning}
              onChange={(e) => setExpectedLearning(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              rows={2}
              placeholder="What would we learn regardless of outcome?"
              required
            />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-xs text-gray-400 mb-1">Intervention Type</label>
              <select
                value={interventionType}
                onChange={(e) => setInterventionType(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
              >
                <option value="config_change">Config Change</option>
                <option value="probe">Probe</option>
                <option value="code_change">Code Change</option>
                <option value="replication">Replication</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Intervention Spec (JSON)</label>
            <textarea
              value={specText}
              onChange={(e) => setSpecText(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm font-mono"
              rows={3}
              placeholder='{"param1": "value1", "param2": "value2"}'
            />
          </div>
          {error && <p className="text-red-400 text-xs">{error}</p>}
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded font-medium disabled:opacity-50"
            >
              {submitting ? "Submitting..." : "Submit Proposal"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

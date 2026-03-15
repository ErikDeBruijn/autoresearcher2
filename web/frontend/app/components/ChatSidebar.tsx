"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

interface ChatSidebarProps {
  onClose: () => void;
}

export default function ChatSidebar({ onClose }: ChatSidebarProps) {
  return (
    <div className="w-96 border-l border-gray-800 flex flex-col bg-gray-950">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-sm font-medium">Research Chat</span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 text-lg leading-none"
        >
          &times;
        </button>
      </div>
      <div className="flex-1 min-h-0">
        <CopilotKit
          runtimeUrl="/api/copilotkit"
          agent="researcher"
          showDevConsole={false}
        >
          <CopilotChat
            labels={{
              title: "Research Assistant",
              initial:
                "Ask me about the research: beliefs, experiments, what to try next, or how the system works.",
              placeholder: "Ask about the research...",
            }}
            className="h-full"
          />
        </CopilotKit>
      </div>
    </div>
  );
}

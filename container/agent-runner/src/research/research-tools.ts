import { defineMcpTool, type McpToolDefinition } from '../mcp-tool-types.js';
import {
  hasResearchWorkspace,
  loadResearchWorkspaceStatus,
} from './workspace-loader.js';

export function createResearchWorkspaceTools(
  workspaceRoot: string,
): McpToolDefinition<any>[] {
  if (!hasResearchWorkspace(workspaceRoot)) return [];

  return [
    defineMcpTool(
      'research_workspace_status',
      'Inspect the current research workspace without modifying it. Returns the workspace and snapshot IDs, manifest completeness, available models, papers, prediction runs, and bounded validation issues. Call this before research tasks so conclusions use the current evidence snapshot.',
      {},
      async () => ({
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify(
              loadResearchWorkspaceStatus(workspaceRoot),
              null,
              2,
            ),
          },
        ],
      }),
    ),
  ];
}

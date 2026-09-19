import { z } from 'zod';

import { defineMcpTool, type McpToolDefinition } from '../mcp-tool-types.js';
import { lookupResearchCode } from './code-lookup.js';
import {
  analyzeResearchPaper,
  audienceResearchPaperAnalysis,
} from './paper-analysis.js';
import { searchResearchPapers } from './paper-search.js';
import { lookupResearchResults } from './result-lookup.js';
import {
  hasResearchWorkspace,
  loadResearchWorkspaceStatus,
} from './workspace-loader.js';

function audiencePaperResult(payload: ReturnType<typeof searchResearchPapers>) {
  return {
    ...payload,
    workspaceId: undefined,
    snapshotId: undefined,
    matches: payload.matches.map((match) => ({
      ...match,
      source: { path: match.source.path, location: match.source.location },
    })),
  };
}

function audienceCodeResult(payload: ReturnType<typeof lookupResearchCode>) {
  return {
    ...payload,
    workspaceId: undefined,
    snapshotId: undefined,
    matches: payload.matches.map((match) => ({
      ...match,
      source: { path: match.source.path, location: match.source.location },
    })),
  };
}

function audienceStructuredResult(
  payload: ReturnType<typeof lookupResearchResults>,
) {
  return { ...payload, internalEvidence: undefined };
}

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
    defineMcpTool(
      'research_paper_analyze',
      'Analyze a paper from its registered paper_id or a relative PDF path inside the current research workspace. Use this first for a new or paper-only input. It returns section-level evidence, analysis capabilities, linked-material status, missing material, and claim boundaries. A provisional PDF can be analyzed without code, configs, logs, weights, or structured results. Do not present missing materials as failures; explain which conclusions remain paper-reported only. Internal hashes and snapshot IDs are returned only when include_internal_evidence is explicitly requested for audit or debugging.',
      {
        paper_id: z.string().trim().min(1).max(100).optional(),
        paper_path: z.string().trim().min(1).max(500).optional(),
        max_sections: z.number().int().min(1).max(12).optional().default(8),
        include_internal_evidence: z.boolean().optional().default(false),
      },
      async (args) => {
        const payload = analyzeResearchPaper(workspaceRoot, {
          paperId: args.paper_id,
          paperPath: args.paper_path,
          maxSections: args.max_sections,
        });
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                args.include_internal_evidence
                  ? payload
                  : audienceResearchPaperAnalysis(payload),
                null,
                2,
              ),
            },
          ],
        };
      },
    ),
    defineMcpTool(
      'research_paper_search',
      'Search prose and method descriptions in text-extractable research papers. Returns bounded page-level excerpts and user-facing citations. Do not use PDF excerpts for table numbers, metrics, parameters, FLOPs, dataset counts, loss weights, or ablation values; call research_result_lookup for those claims. Internal snapshot and hash fields are returned only when include_internal_evidence is explicitly requested for audit or debugging.',
      {
        query: z.string().trim().min(2).max(200),
        paper_id: z.string().trim().min(1).max(100).optional(),
        max_results: z.number().int().min(1).max(10).optional().default(5),
        include_internal_evidence: z.boolean().optional().default(false),
      },
      async (args) => {
        const payload = searchResearchPapers(workspaceRoot, {
          query: args.query,
          paperId: args.paper_id,
          maxResults: args.max_results,
        });
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                args.include_internal_evidence
                  ? payload
                  : audiencePaperResult(payload),
                null,
                2,
              ),
            },
          ],
        };
      },
    ),
    defineMcpTool(
      'research_result_lookup',
      'Read row-bound quantitative facts from structured paper results. Use this tool for every metric, parameter count, FLOPs value, dataset count, loss weight, comparison, or ablation claim. Each fact is an atomic row: never combine a model or scene with values from another row. Internal hashes and snapshot IDs are returned only when include_internal_evidence is explicitly requested for audit or debugging.',
      {
        paper_id: z.string().trim().min(1).max(100).optional(),
        table: z
          .enum([
            'all',
            'dataset_counts',
            'loss_weights',
            'comparison',
            'complexity',
            'ablation',
          ])
          .optional()
          .default('all'),
        model: z.string().trim().min(1).max(100).optional(),
        scene: z.string().trim().min(1).max(100).optional(),
        variant: z.string().trim().min(1).max(100).optional(),
        max_results: z.number().int().min(1).max(50).optional().default(20),
        include_internal_evidence: z.boolean().optional().default(false),
      },
      async (args) => {
        const payload = lookupResearchResults(workspaceRoot, {
          paperId: args.paper_id,
          table: args.table,
          model: args.model,
          scene: args.scene,
          variant: args.variant,
          maxResults: args.max_results,
        });
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                args.include_internal_evidence
                  ? payload
                  : audienceStructuredResult(payload),
                null,
                2,
              ),
            },
          ],
        };
      },
    ),
    defineMcpTool(
      'research_code_lookup',
      'Search only source_code files registered in the current research snapshot. Returns bounded source excerpts with model, symbol, line numbers, and user-facing relative paths. Use this tool to locate implementations and calls; cite only returned lines and do not infer that a name match proves runtime behavior. Internal hashes and snapshot IDs are returned only when include_internal_evidence is explicitly requested for audit or debugging.',
      {
        query: z.string().trim().min(1).max(200),
        model: z.string().trim().min(1).max(100).optional(),
        max_results: z.number().int().min(1).max(20).optional().default(10),
        include_internal_evidence: z.boolean().optional().default(false),
      },
      async (args) => {
        const payload = lookupResearchCode(workspaceRoot, {
          query: args.query,
          model: args.model,
          maxResults: args.max_results,
        });
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                args.include_internal_evidence
                  ? payload
                  : audienceCodeResult(payload),
                null,
                2,
              ),
            },
          ],
        };
      },
    ),
  ];
}

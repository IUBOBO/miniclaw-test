export type ResearchSourceType =
  | 'paper'
  | 'code'
  | 'config'
  | 'log'
  | 'metadata'
  | 'prediction'
  | 'sample';

export type EvidenceStatus = 'verified' | 'unknown' | 'conflict';

export interface EvidenceRef {
  sourceType: ResearchSourceType;
  /** Path relative to the research workspace root. */
  path: string;
  /** Fixed research workspace snapshot. */
  snapshotId: string;
  /** SHA-256 from snapshot-manifest.jsonl when available. */
  sha256?: string;
  page?: number;
  table?: number;
  lineStart?: number;
  lineEnd?: number;
  symbol?: string;
  status: EvidenceStatus;
}

export interface ResearchFinding {
  id: string;
  question: string;
  conclusion: string;
  evidence: EvidenceRef[];
  unknowns: string[];
  conflicts: string[];
}

export type ResearchWorkspaceHealth = 'ready' | 'degraded' | 'error';
export type ResearchIssueSeverity = 'warning' | 'error';

export interface ResearchIssue {
  code:
    | 'WORKSPACE_CONFIG_INVALID'
    | 'RESEARCH_ROOT_INVALID'
    | 'MANIFEST_INVALID'
    | 'MANIFEST_DUPLICATE_PATH'
    | 'MANIFEST_FILE_MISSING'
    | 'MANIFEST_SIZE_MISMATCH'
    | 'CONTROL_HASH_MISMATCH'
    | 'METADATA_INVALID'
    | 'SNAPSHOT_MISMATCH'
    | 'PATH_OUTSIDE_WORKSPACE'
    | 'READ_FAILED';
  severity: ResearchIssueSeverity;
  message: string;
  path?: string;
}

export interface ResearchWorkspaceStatus {
  status: ResearchWorkspaceHealth;
  workspaceId: string | null;
  snapshotId: string | null;
  manifest: {
    total: number;
    present: number;
    missing: number;
    controlHashesChecked: number;
    controlHashMismatches: number;
  };
  models: string[];
  papers: string[];
  predictionRuns: string[];
  issueCount: number;
  issues: ResearchIssue[];
}

export const CLAUDE_ENDPOINT_KIND_ENV = 'MINICLAW_CLAUDE_ENDPOINT_KIND';
export const MINICLAW_CLAUDE_ENDPOINT_KIND_ENV =
  'MINICLAW_CLAUDE_ENDPOINT_KIND';

/**
 * Provider 使用的 API 协议。
 *
 * anthropic-messages:
 *   Anthropic Messages API 兼容协议
 *
 * openai-completions:
 *   OpenAI Chat Completions 兼容协议
 */
export const MINICLAW_PROVIDER_API_ENV = 'MINICLAW_PROVIDER_API';

export type ClaudeEndpointKind = 'official' | 'custom';

export type ProviderApiProtocol =
  | 'anthropic-messages'
  | 'openai-completions';

export interface ClaudeProviderRuntime {
  endpointKind: ClaudeEndpointKind;

  /**
   * 底层模型接口协议。
   *
   * 官方 Claude 固定使用 anthropic-messages。
   * 第三方 Provider 可以显式选择 Anthropic 或 OpenAI Compatible。
   */
  api: ProviderApiProtocol;

  model: string;
  queryModelOptions: { model?: string };
  usageModelKey: string;
  missingRequiredModel: boolean;
}

export interface ClaudeQueryModelRuntime {
  model: string;
  queryModelOptions: { model?: string };
  usageModelKey: string;
}

/** Resolve a model at query time so a warm runner can switch tiers without respawn. */
export function resolveClaudeQueryModelRuntime(
  providerRuntime: ClaudeProviderRuntime,
  modelOverride?: string,
): ClaudeQueryModelRuntime {
  const model = modelOverride?.trim() || providerRuntime.model;

  return {
    model,
    queryModelOptions: model ? { model } : {},
    usageModelKey: model || 'default',
  };
}

function resolveProviderApi(
  env: Readonly<Record<string, string | undefined>>,
  endpointKind: ClaudeEndpointKind,
): ProviderApiProtocol {
  /**
   * 官方 Claude 不允许切换协议。
   */
  if (endpointKind === 'official') {
    return 'anthropic-messages';
  }

  const configuredApi = env[MINICLAW_PROVIDER_API_ENV]
    ?.trim()
    .toLowerCase();

  if (configuredApi === 'openai-completions') {
    return 'openai-completions';
  }

  if (configuredApi === 'anthropic-messages') {
    return 'anthropic-messages';
  }

  /**
   * 兼容旧版 Miniclaw。
   *
   * 以前第三方 Provider 没有协议字段，
   * 并且统一按 Anthropic Messages API 调用。
   * 因此没有设置 MINICLAW_PROVIDER_API 时继续保持旧行为。
   */
  return 'anthropic-messages';
}

/**
 * Resolve the provider/model contract once at runner startup.
 *
 * New hosts inject an authoritative endpoint-kind marker. Falling back to
 * ANTHROPIC_BASE_URL keeps the runner compatible with older hosts and images.
 */
export function resolveClaudeProviderRuntime(
  env: Readonly<Record<string, string | undefined>>,
): ClaudeProviderRuntime {
  const model = env.ANTHROPIC_MODEL?.trim() ?? '';

  const marker = (
    env[MINICLAW_CLAUDE_ENDPOINT_KIND_ENV] ||
    env[CLAUDE_ENDPOINT_KIND_ENV]
  )
    ?.trim()
    .toLowerCase();

  const endpointKind: ClaudeEndpointKind =
    marker === 'official' || marker === 'custom'
      ? marker
      : env.ANTHROPIC_BASE_URL?.trim()
        ? 'custom'
        : 'official';

  const api = resolveProviderApi(env, endpointKind);

  return {
    endpointKind,
    api,
    model,
    queryModelOptions: model ? { model } : {},
    usageModelKey: model || 'default',
    missingRequiredModel: endpointKind === 'custom' && !model,
  };
}
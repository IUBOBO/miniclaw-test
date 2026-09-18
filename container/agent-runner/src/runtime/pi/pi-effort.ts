import { resolveAgentSdkEffort } from '../../agent-effort.js';
import type { RuntimeSessionOptions } from '../types.js';

export function resolvePiEffort(
  policy: unknown,
  env: Readonly<Record<string, string | undefined>>,
): RuntimeSessionOptions['thinkingLevel'] {
  return (
    resolveAgentSdkEffort(policy) ??
    resolveAgentSdkEffort({
      reasoning: { effort: env.CLAUDE_CODE_EFFORT_LEVEL },
    })
  );
}

// This gateway's V4.1 endpoint requires chat_template_kwargs rather than a
// generic top-level OpenAI reasoning_effort. Keep the exception endpoint-scoped.
export function usesDeepSeekV41Template(
  baseUrl: string,
  model: string,
): boolean {
  try {
    return (
      new URL(baseUrl).hostname === 'api.icompify.com' &&
      /(?:^|\/)deepseek-v4\.1(?:-|$)/i.test(model)
    );
  } catch {
    return false;
  }
}

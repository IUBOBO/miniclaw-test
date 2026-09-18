import { describe, expect, test, vi } from 'vitest';

import {
  resolvePiEffort,
  usesDeepSeekV41Template,
} from '../container/agent-runner/src/runtime/pi/pi-effort.js';
import { resolvePiProvider } from '../container/agent-runner/src/runtime/pi/pi-provider.js';

describe('Pi provider compatibility', () => {
  test('forwards explicit Agent effort and falls back to Provider effort', () => {
    expect(resolvePiEffort({ reasoning: { effort: 'high' } }, {})).toBe('high');
    expect(
      resolvePiEffort(
        { reasoning: { effort: 'inherit' } },
        { CLAUDE_CODE_EFFORT_LEVEL: 'low' },
      ),
    ).toBe('low');
    expect(resolvePiEffort(undefined, {})).toBeUndefined();
  });

  test('limits the DeepSeek template override to the verified endpoint', () => {
    expect(
      usesDeepSeekV41Template(
        'https://api.icompify.com/v1',
        'deepseek-v4.1-flash',
      ),
    ).toBe(true);
    expect(
      usesDeepSeekV41Template(
        'https://another.example/v1',
        'deepseek-v4.1-flash',
      ),
    ).toBe(false);
  });

  test('registers the gateway model with chat_template_kwargs reasoning', async () => {
    let registered: any;
    const modelRuntime = {
      registerProvider: vi.fn((_id: string, provider: unknown) => {
        registered = provider;
      }),
      getModel: vi.fn((_providerId: string, modelId: string) => ({
        ...registered.models[0],
        id: modelId,
      })),
    };

    await resolvePiProvider(modelRuntime as any, {
      model: 'deepseek-v4.1-flash',
      endpointKind: 'custom',
      api: 'openai-completions',
      baseUrl: 'https://api.icompify.com/v1',
      apiKey: 'test-only',
    });

    expect(registered.models[0]).toMatchObject({
      api: 'openai-completions',
      thinkingLevelMap: { medium: 'low', high: 'high' },
      compat: {
        thinkingFormat: 'chat-template',
        chatTemplateKwargs: {
          reasoning_effort: { $var: 'thinking.effort' },
        },
      },
    });
  });
});

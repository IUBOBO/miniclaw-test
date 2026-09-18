import { describe, expect, test } from 'vitest';

import {
  checkHostCapabilities,
  resetHostCapabilitiesCache,
} from '../src/agent-capabilities.js';

describe('host agent capability preflight', () => {
  test('finds the Node executable with the platform-native command locator', async () => {
    resetHostCapabilitiesCache();

    const result = await checkHostCapabilities();

    expect(
      result.available.some((capability) => capability.binary === 'node'),
    ).toBe(true);
    expect(result.resolvedPaths.node).toBeTruthy();
  });
});

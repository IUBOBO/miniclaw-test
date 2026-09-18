import { describe, expect, test } from 'vitest';

import { providerRequestErrorNotice } from '../src/provider-request-error.js';

describe('provider request error notice', () => {
  test('explains incompatible reasoning effort without exposing upstream JSON', () => {
    expect(
      providerRequestErrorNotice('400: reasoning_effort must be low or high'),
    ).toContain('推理档位');
  });

  test('explains protocol URL and stream mismatches', () => {
    expect(providerRequestErrorNotice('404 404 page not found')).toContain(
      'Base URL',
    );
    expect(
      providerRequestErrorNotice(
        'Anthropic stream ended without a stop reason',
      ),
    ).toContain('协议');
  });

  test('keeps transient errors retryable', () => {
    expect(providerRequestErrorNotice('429 rate limit')).toBeNull();
    expect(providerRequestErrorNotice('connection reset')).toBeNull();
  });
});

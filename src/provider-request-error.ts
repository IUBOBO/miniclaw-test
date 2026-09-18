const CLIENT_ERROR_STATUS_RE = /^(?:400|404)(?:\s|:)/;

export function providerRequestErrorNotice(error: string): string | null {
  if (/reasoning_effort/i.test(error)) {
    return '模型服务拒绝了推理档位参数。请检查 Agent 推理档位与该模型的兼容配置。';
  }
  if (/404 page not found/i.test(error) || /^404(?:\s|:)/.test(error)) {
    return '模型接口返回 404。请检查 Provider 协议和 Base URL 是否匹配。';
  }
  if (/stream ended without a stop reason/i.test(error)) {
    return '模型流式响应未正常结束。请检查 Provider 协议是否与接口返回格式一致。';
  }
  if (CLIENT_ERROR_STATUS_RE.test(error)) {
    return '模型服务拒绝了本次请求。请检查模型名称、接口协议和请求参数。';
  }
  return null;
}

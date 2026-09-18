# Provider 协议切换使用说明

适用版本：V0.1.3。项目位置：E:/Agent/miniclaw-test。

## 使用步骤

1. 浏览器刷新后，打开设置中的模型配置，编辑第三方 Provider。
2. 选择 Anthropic Compatible 或 OpenAI Compatible。
3. 第一次选择某协议时填写服务商提供的 Base URL；以后会自动恢复该协议已保存的地址，也可以直接修改。
4. 核对模型名和密钥，点击保存。保存后新请求使用所选协议及对应地址；不需要手动重启服务。
5. 发送一条新消息验证。历史错误保留在聊天记录中，不代表新请求仍然失败。

| 选择 | 请求协议 | 地址填写原则 |
|---|---|---|
| Anthropic Compatible | Anthropic Messages | 使用服务商提供的 Anthropic 基础地址 |
| OpenAI Compatible | OpenAI Chat Completions | 使用服务商提供的 OpenAI 基础地址，常见为 /v1 |

地址完全由用户填写，不根据模型名自动改协议，也不猜测或自动拼接服务商的基础地址。切换到未配置协议时不会借用另一协议地址。

同一 Provider 的两种协议独立保存地址，模型名、密钥和高级参数共用。如果两种接口需要不同的模型、账号或密钥，请建立两条 Provider 并在 Agent 中选择对应配置。官方 Claude 仅支持 Anthropic。

## 实现与验收

- protocolBaseUrls 保存两种地址；apiProtocol 决定当前生效协议，兼容字段 anthropicBaseUrl 表示当前生效地址。
- 旧数据读取时将单一地址归入原协议；不会替用户填写未知的另一地址。
- 修改当前地址或协议会停止相关旧运行并使绑定会话失效，下一次请求采用新配置。
- 4 个测试文件、43 项测试通过；后台、网页和 Runner 构建通过。自动测试不代替真实服务商连通性验证。
- 科研工具链及真实工作区文件读取仍需单独验收。

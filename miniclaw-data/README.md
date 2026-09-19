# BDGI-Net Research Materials

本目录用于 MiniClaw 科研 Agent 的证据检索、源码定位和结论核验。当前快照：`SNAP-20260919-92afd6116c4d`。

## 证据优先级

1. `PDFpaper/BDGI-Net.pdf`：数据集规模、最终指标、消融与复杂度的最终来源。
2. `mymodel/code/`：证明论文模块、损失与评价算法存在。
3. `samples/evaluations/`：山东和四川保存掩码的可核对样例，不覆盖论文最终指标。
4. 交接文档：仅用于目录和运行入口说明。

## 目录

- `mymodel/`：BDGI-Net 源码、历史配置与日志。
- `othermodel/`：HBGNet、REAUNet、SEANet、BsiNet 材料；TFNet 不在本轮范围。
- `PDFpaper/`：BDGI-Net 与对比论文 PDF。
- `samples/evaluations/SD/`：73 组山东输入、真实掩码、保存预测。
- `samples/evaluations/SC/`：20 组四川输入、真实掩码、保存预测。
- `metadata/`：论文、结果、指标、Prediction Run 与 Ownership 元数据。
- `notes/methods/BDGI-Net.md`：论文—源码映射与证据边界。

## 使用边界

最终性能只引用 `metadata/results/PAPER-BDGI-RESULTS.yaml`。保存预测的单图核对值只用于案例说明。本材料不声明完成训练复现、权重复现、历史划分复现或边界预测评测。

`snapshot-manifest.jsonl` 记录所有纳管文件的来源、大小、修改时间和 SHA-256；清单自身不纳入自身哈希。

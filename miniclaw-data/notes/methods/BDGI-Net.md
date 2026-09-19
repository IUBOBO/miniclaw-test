---
paper_id: PAPER-BDGI
model: BDGI-Net
status: source_verified
source_snapshot_id: SNAP-20260919-92afd6116c4d
updated_at: 2026-09-19T05:50:28.266018+00:00
---

# BDGI-Net 论文—源码映射

本文档以论文为最终结论来源，以源码证明实现存在。行号对应当前快照；源码变化后必须重新核对。

| 论文概念 | 论文位置 | 源码位置 | SHA-256 | 核验结论 |
|---|---|---|---|---|
| BDGI-Net | PDF 第 1–5 页，方法总览 | `mymodel/code/model.py:408`，`Tripmodel`；前向路径 `459–505` | `1fb28fb87b3f846b9d2f7743eb0ec291fbabe73126bbf4aba835031a4915a39d` | 共享编码后分别构造语义/边界特征，最终返回 mask 与 boundary。 |
| ETA | PDF 第 3–4 页 | `mymodel/code/model.py:180`，`EnhancedTripletAttention`；调用 `421–424` | `1fb28fb87b3f846b9d2f7743eb0ec291fbabe73126bbf4aba835031a4915a39d` | 在多层特征上执行增强三重注意力。 |
| TAB | PDF 第 4–5 页 | `mymodel/code/model.py:298`，`TokenAggregationBlock`；调用 `427–428` | `1fb28fb87b3f846b9d2f7743eb0ec291fbabe73126bbf4aba835031a4915a39d` | 组合 LocalTokenAttention 与 GlobalTokenAttention。 |
| GMRM | PDF 第 5 页 | `mymodel/code/model.py:332`，`MaskGuidedByBoundaryModule`；调用 `443`、`500` | `1fb28fb87b3f846b9d2f7743eb0ec291fbabe73126bbf4aba835031a4915a39d` | 从 boundary feature 生成引导并细化 mask feature。 |
| 双分支解码 | PDF 第 1–5 页 | `mymodel/code/model.py:449–505`，`boundary_head`、`mask_head` | `1fb28fb87b3f846b9d2f7743eb0ec291fbabe73126bbf4aba835031a4915a39d` | 掩码与边界分别输出，边界特征在掩码输出前参与引导。 |
| 多任务损失 | PDF 第 6–7 页 | `mymodel/code/LossFunction.py:41–79`，`BCEDiceLoss`、`MultiTaskLoss` | `d1e84225566adddba2af589984495527f6259e6f09abce7435e455347860b2f7` | 生效实现采用 mask 0.6、boundary 0.4。 |
| 像素掩码指标 | PDF 第 7 页，式 13–14 | `mymodel/code/mask_ARF.py:18–35`、`38–49`、`107–125` | `c191253e5d5292ca108f11b9cd0868cda98740bd56c467bfcc3ad33c66448b61` | 逐图计算 Precision/Recall/F1/IoU/Accuracy 后算术平均。 |
| 对象掩码指标 | PDF 第 7 页，式 15–20 | `mymodel/code/mask_eval.py:7–129` | `6f01cb230559191da30c304ae380531146264b34422f66ce7493fdbedb034baf` | 连通域匹配后按预测地块面积汇总 GOC/GUC/GTC。 |

## 论文结果导航

- 表 1（数据集规模）：PDF 第 7 页。
- 表 2（损失权重）：PDF 第 7 页。
- 表 3（三场景对比）：PDF 第 8 页。
- 表 4（参数量与 FLOPs）：PDF 第 9 页。
- 表 7（消融实验）：PDF 第 10 页。
- 结构化结果：`metadata/results/PAPER-BDGI-RESULTS.yaml`。

## 证据边界

- 论文数据规模与最终指标直接按论文记录，不要求用本地样例重算复现。
- SD/SC 保存预测只证明现有掩码结果可核对，不能覆盖论文表 3。
- `mask_ARF.py` 与 `mask_eval.py` 分别负责像素指标和对象指标，不视为冲突实现。
- 本轮不审计训练划分、训练环境、最终权重绑定、边界预测、TFNet 或数据许可。

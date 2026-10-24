# 数据集选择：团购与搜推广告 OPD

2026-10-24。主实验调整为公开业务数据。原餐饮、酒店、到店服务模拟任务用于补充约束检查；本项目尚无 GPU 训练收益。

| 数据集 | 业务任务 | Teacher 与指标 | 选型 |
|---|---|---|---|
| Amazon ESCI | 搜索相关性、重排、替代品识别 | 强 LLM / cross-encoder；NDCG@K、MRR、macro-F1 | 主实验，有真实 query/商品文本和人工标签 |
| 天池 O2O 优惠券 | 领券后 15 天核销预测 | 时间隔离 GBDT；按券平均 AUC、PR-AUC、LogLoss、校准 | 团购扩展，下载与条款待核对 |
| KuaiRand | 推荐、点击/有效播放、多目标偏好 | 专用推荐 teacher；AUC、NDCG、长观看率 | 推荐扩展，语义补充映射待核对 |
| Ali-CCP / Criteo | CTR/CVR / 广告点击预测 | 专用多任务模型；AUC、LogLoss、ECE | 广告扩展，匿名特征不假定为 LLM 语义 |

优先完成 ESCI 的 LLM OPD 闭环，比较标签 SFT、离线蒸馏、GRPO、OPD、混合训练和 cross-encoder。保留数学/指令回退和同 H20 上的延迟/成本测量。ESCI 是商品搜索；O2O 是核销代理任务，均不等价真实团购成交收益。

ESCI 沿用官方 train/test；dev 仅从 train 划出。归一化 query 分组去重，报告商品重叠与语言切片。Prompt 不含标签，固定候选数及采样，不挑 teacher 占优样本。O2O 按时间划分并保留 15 天标签成熟期，特征仅使用领取时刻前的历史。

KuaiSAR 的 keyword/caption 已逐词哈希，不当作原始自然语言。KuaiRand 随机曝光可辅助偏差分析，但完整 OPE 仍需核对 propensity。离线日志成绩不能声称线上 CTR/CVR 因果收益。

固定更新数、有效 response tokens、GPUh 分别比较，不宣称长度不同的方法同时严格匹配更新数和 tokens。公开仓库提供来源、处理脚本、划分 ID、哈希及独立数据条款，扩展候选未下载/训练则不视为就绪。

来源：

- [ESCI 官方](https://github.com/amazon-science/esci-data)：字段、人工标签、官方划分、Apache-2.0 项目许可。
- [ESCI 论文](https://arxiv.org/abs/2206.06588)。
- [天池 O2O 官方 notebook](https://tianchi.aliyun.com/notebook/22891)：15 天核销目标。
- [KuaiRand 官方](https://kuairand.com/)：随机曝光、反馈及语义补充入口。
- [KuaiSAR 官方字段](https://kuaisar.github.io/detailed_statistics.html)：keyword/caption 哈希。
- [Criteo 官方条款](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)：CC BY-NC-SA 4.0；研究许可不等同商业部署授权。

# SCP-Flow 方法说明

最后更新：2026-03-28

本文档描述当前仓库中 `SCP-Flow` 的实际实现版本，对应代码主要位于：

- [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py)
- [models/flow_forecaster.py](/d:/desk/fm4glaucoma/models/flow_forecaster.py)
- [datamodule/seq_fundus_2D_datamodule.py](/d:/desk/fm4glaucoma/datamodule/seq_fundus_2D_datamodule.py)

注意：

- 这里描述的是当前已经落地的 `minimum runnable SCP-Flow`
- 不是目标论文中的最终理想版本

## 一、任务定义

给定一个 SIGF clip：

$$
\mathcal{S}=\{(x_1,l_1,t_1),\dots,(x_T,l_T,t_T)\}
$$

其中：

- $x_i$：第 $i$ 次随访图像
- $l_i$：第 $i$ 次随访标签
- $t_i$：第 $i$ 次随访时间

当前实现中，单个 clip 长度固定为：

$$
T=6
$$

模型使用前 $T-1$ 次历史去预测第 $T$ 次未来状态。为避免未来信息泄漏，送入条件编码器的最后一个位置会被替换为第 $T-1$ 次观测。

## 二、模型输入与输出

### 2.1 数据输入

当前 dataloader 输出的单个样本包括：

- `image`：完整图像序列，形状近似为 $(T,3,H,W)$
- `time`：离散时间序列，形状为 $(T,)$
- `label`：标签序列，形状为 $(T,)$
- `image_id`：样本 ID
- `disc_roi`：从最后一次可见图像中裁出的视盘区域
- `polar_roi`：对应的极坐标 ROI
- `vcdr`：启发式估计的垂直杯盘比
- `ocod`：启发式估计的 cup/disc 面积比代理量
- `missing_mask`：缺访模拟掩码

其中结构条件由 [seq_fundus_2D_datamodule.py](/d:/desk/fm4glaucoma/datamodule/seq_fundus_2D_datamodule.py) 动态构造。

### 2.2 训练阶段内部输入

在 [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py) 的 `get_input(...)` 中，实际训练使用：

- `x_seq`：条件图像序列
- `t_seq`：条件时间序列
- `l_seq`：条件标签序列
- `prev_latent = z_{T-1}`
- `prev_prev_latent = z_{T-2}`
- `target_latent = z_T`
- `target_interval = t_T - t_{T-1}`
- `struct_cond`

其中 latent 由预训练 VQGAN 编码器得到：

$$
z_i = E(x_i)
$$

这里的 $E(\cdot)$ 是冻结的第一阶段编码器。

### 2.3 模型输出

当前 `ProgressionFlowForecaster` 输出：

- `velocity`：$\hat v$
- `pred_target`：$\hat z_T$
- `pred_interval`：$\widehat{\Delta t}$
- `pred_uncertainty`：$\hat \sigma$
- `pred_structure`：$\hat s$

解码后还可得到未来图像预测：

$$
\hat x_T = D(\hat z_T)
$$

其中 $D(\cdot)$ 是冻结的第一阶段解码器。

## 三、整体架构

当前架构可以分为 4 个模块。

### 3.1 第一阶段图像自编码器

来自原始 `tHPM-LDM`，当前继续复用：

- 编码器 $E$
- 解码器 $D$

作用是将未来预测问题从像素空间转到 latent 空间：

$$
z_i = E(x_i), \qquad \hat x_T = D(\hat z_T)
$$

### 3.2 历史条件编码器

复用原始 `ConditionGenMSTFCMPopuMemory`，输出为：

$$
(c_h, c_p) = \mathrm{CondGen}(x_{1:T-1}, t_{1:T-1}, l_{1:T-1})
$$

其中：

- $c_h$：个体历史条件
- $c_p$：群体记忆条件

当前代码中对应：

- `ch`
- `cp`

### 3.3 结构条件编码器

当前结构条件分为两部分：

1. 标量结构条件：

$$
s_{\text{scalar}} = [\mathrm{vCDR}, \mathrm{OCOD}]
$$

2. ROI 图像结构条件：

- `disc_roi`
- `polar_roi`

在 [flow_forecaster.py](/d:/desk/fm4glaucoma/models/flow_forecaster.py) 中，结构编码器将它们编码为：

$$
c_s = \mathrm{StructEnc}(s_{\text{scalar}}, r_{\text{disc}}, r_{\text{polar}})
$$

更具体地写为：

$$
c_s = \mathrm{LN}\Big(
\phi_{\text{scalar}}([\mathrm{vCDR},\mathrm{OCOD}])
 + \phi_{\text{disc}}(r_{\text{disc}})
 + \phi_{\text{polar}}(r_{\text{polar}})
\Big)
$$

其中：

- $\phi_{\text{scalar}}$：两层 MLP
- $\phi_{\text{disc}}$：卷积 ROI encoder
- $\phi_{\text{polar}}$：卷积 ROI encoder
- $\mathrm{LN}$：LayerNorm

### 3.4 Progression Flow Forecaster

这是当前 `SCP-Flow` 的核心模块。

给定桥接中间状态 $z_\tau$、连续时间 $\tau$、历史条件 $c_h$、群体条件 $c_p$、结构条件 $c_s$ 和上一时刻 latent $z_{T-1}$，模型输出速度场和多个预测头：

$$
(\hat v, \hat z_T, \widehat{\Delta t}, \hat\sigma, \hat s)
=
F_\theta(z_\tau,\tau,c_h,c_p,c_s,z_{T-1})
$$

在实现上，它包含：

1. `TauEmbedding`

$$
c_\tau = \phi_\tau(\tau)
$$

2. `prev_latent_pool`

$$
c_{\text{prev}} = \phi_{\text{prev}}(z_{T-1})
$$

3. 条件融合器

$$
c = \phi_{\text{fuse}}([c_h,c_p,c_\tau,c_{\text{prev}},c_s])
$$

4. Flow 主干

输入是当前中间 latent 与上一时刻 latent 的通道拼接：

$$
h_0 = \mathrm{Conv}([z_\tau, z_{T-1}])
$$

然后经过 3 个 FiLM 残差块：

$$
h_{k+1} = \mathrm{FiLMResBlock}(h_k, c)
$$

最后得到速度预测：

$$
\hat v = \mathrm{Conv}_{\text{vel}}(h_3)
$$

并根据当前实现中的显式重参数化，得到目标 latent：

$$
\hat z_T = z_\tau + (1-\tau)\hat v
$$

另外从全局条件向量 $c$ 上接三个头：

$$
\widehat{\Delta t} = \phi_t(c)
$$

$$
\hat \sigma = \mathrm{softplus}(\phi_\sigma(c)) + 10^{-4}
$$

$$
\hat s = \phi_s(c)
$$

## 四、桥接采样与 flow 学习

当前训练不是直接输入 $z_{T-1}$ 去预测 $z_T$，而是在二者之间采样一个桥接中间状态。

### 4.1 目标 latent

$$
z_{T-1} = E(x_{T-1}), \qquad z_T = E(x_T)
$$

### 4.2 中间时间采样

训练时随机采样：

$$
\tau \sim \mathcal U(0,1)
$$

实际实现中裁剪为：

$$
\tau \in [10^{-3}, 1-10^{-3}]
$$

### 4.3 桥接中间状态

当前桥接分布使用高斯桥近似：

$$
\mu_\tau = (1-\tau)z_{T-1} + \tau z_T
$$

$$
\sigma_\tau = \sigma_b \sqrt{\tau(1-\tau)}
$$

$$
z_\tau = \mu_\tau + \sigma_\tau \epsilon,\qquad \epsilon \sim \mathcal N(0,I)
$$

其中 $\sigma_b$ 对应代码中的 `flow_bridge_sigma`。

### 4.4 监督速度

当前实现中，目标速度采用终点差：

$$
v^\star = z_T - z_{T-1}
$$

然后用 MSE 拟合：

$$
\mathcal L_{\text{vel}} = \|\hat v - v^\star\|_2^2
$$

## 五、损失函数

总损失由 6 到 7 项组成：

$$
\mathcal L =
\lambda_v \mathcal L_{\text{vel}}
 + \lambda_z \mathcal L_{\text{target}}
 + \lambda_x \mathcal L_{\text{recon}}
 + \lambda_t \mathcal L_{\text{interval}}
 + \lambda_u \mathcal L_{\text{uncertainty}}
 + \lambda_c \mathcal L_{\text{consistency}}
 + \lambda_s \mathcal L_{\text{structure}}
$$

对应权重分别来自：

- `flow_velocity_weight`
- `flow_target_weight`
- `flow_recon_weight`
- `interval_loss_weight`
- `uncertainty_loss_weight`
- `consistency_loss_weight`
- `structure_loss_weight`

### 5.1 目标 latent 损失

$$
\mathcal L_{\text{target}} = \|\hat z_T - z_T\|_2^2
$$

### 5.2 图像重建损失

先解码未来预测 latent：

$$
\hat x_T = D(\hat z_T)
$$

然后对真实未来图像做 L1：

$$
\mathcal L_{\text{recon}} = \|\hat x_T - x_T\|_1
$$

### 5.3 时间间隔损失

$$
\Delta t^\star = t_T - t_{T-1}
$$

$$
\mathcal L_{\text{interval}} = (\widehat{\Delta t} - \Delta t^\star)^2
$$

### 5.4 不确定性损失

当前实现中先定义样本级 latent 残差：

$$
r = \mathrm{mean}\big((\hat z_T - z_T)^2\big)
$$

然后使用异方差形式目标：

$$
\mathcal L_{\text{uncertainty}} = \frac{r}{\hat\sigma} + \log \hat\sigma
$$

整体对 batch 取平均。

### 5.5 病程一致性损失

定义预测病程增量：

$$
\Delta z_{\text{pred}} = \mathrm{mean}_{h,w}(\hat z_T - z_{T-1})
$$

定义历史病程增量：

$$
\Delta z_{\text{hist}} = \mathrm{mean}_{h,w}(z_{T-1} - z_{T-2})
$$

则一致性损失为：

$$
\mathcal L_{\text{consistency}}
=
\|\Delta z_{\text{pred}} - \Delta z_{\text{hist}}\|_2^2
$$

### 5.6 结构监督损失

当前结构头预测的是 2 维结构向量：

$$
\hat s = [\widehat{\mathrm{vCDR}}, \widehat{\mathrm{OCOD}}]
$$

监督目标为 dataloader 中提供的结构代理值：

$$
s^\star = [\mathrm{vCDR}, \mathrm{OCOD}]
$$

损失定义为：

$$
\mathcal L_{\text{structure}} = \|\hat s - s^\star\|_2^2
$$

注意：

- 这项损失只有在 batch 中存在 `vcdr` 和 `ocod` 时才启用
- 当前这两个值是启发式代理值，不是人工标注

## 六、推理过程

测试阶段不再随机采样桥接状态，而是直接令：

$$
\tau = 0,\qquad z_\tau = z_{T-1}
$$

即直接从上一时刻 latent 出发：

$$
(\hat v, \hat z_T, \widehat{\Delta t}, \hat\sigma, \hat s)
=
F_\theta(z_{T-1},0,c_h,c_p,c_s,z_{T-1})
$$

然后解码出预测图像：

$$
\hat x_T = D(\hat z_T)
$$

测试阶段导出：

- `*_gen.png`
- `*_gt.png`
- `*_rec.png`
- `flow_predictions.csv`

其中 `flow_predictions.csv` 当前包含：

- `case_id`
- `pred_interval`
- `target_interval`
- `pred_uncertainty`
- `missing_count`
- `missing_strategy`
- `abs_interval_error`

## 七、缺访建模

当前缺访不是通过 mask token 学习，而是在 dataloader 层进行模拟。

给定历史 visits 索引集合：

$$
\{1,\dots,T-1\}
$$

按照策略选取缺失索引集合 $\mathcal M$：

- `tail`
- `uniform`
- `random`

对于被删掉的历史 visit，当前实现采用：

- 若缺的是第一个历史点，则置零图像并置零标签
- 否则用前一个 visit 的图像、时间和标签复制填充

因此当前缺访评估更接近“信息退化鲁棒性测试”，而不是严格的可学习 mask-based missing modeling。

## 八、当前方法的实现边界

当前 `SCP-Flow` 已实现：

- history-conditioned flow 预测
- `ch/cp` 条件接入
- 结构条件接入
- 下一次随访间隔预测
- 不确定性预测
- 病程一致性正则
- 缺访鲁棒性评估

当前还未实现或未完善：

- 多步 rollout
- 真实结构标注监督
- 强不确定性校准
- 显式结构可控实验
- 与原始 diffusion 的正式统一 benchmark

## 九、代码对应关系速查

### 输入构造

- [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py) 中的 `get_input(...)`
- [datamodule/seq_fundus_2D_datamodule.py](/d:/desk/fm4glaucoma/datamodule/seq_fundus_2D_datamodule.py)

### 条件生成

- [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py) 中的 `_get_conditions(...)`
- `ConditionGenMSTFCMPopuMemory`

### 桥接采样

- [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py) 中的 `_sample_bridge(...)`

### 主损失

- [train_vqldm_PMN.py](/d:/desk/fm4glaucoma/train_vqldm_PMN.py) 中的 `_forward_losses(...)`

### Flow 主干

- [models/flow_forecaster.py](/d:/desk/fm4glaucoma/models/flow_forecaster.py)

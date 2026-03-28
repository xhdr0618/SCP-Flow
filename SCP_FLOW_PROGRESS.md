# SCP-Flow 项目进度

最后更新：2026-03-28

## 一、项目目标

配套文档：

- 方法文档：`SCP_FLOW_METHODS.md`

基于当前 `tHPM-LDM` 仓库，尽量少破坏原有结构，将未来预测主干从：

- 条件 latent diffusion 预测器

改造成：

- 条件 trajectory flow / progression flow 预测器

并在保留原始优势的基础上新增以下能力：

- 结构条件控制：
  - `vCDR`
  - `OC-OD`
  - `disc ROI`
  - `polar ROI`
- 下一次随访时间间隔预测
- 不确定性预测
- 病程一致性正则
- 缺访鲁棒性评估

## 二、当前总体状态

当前阶段：`SCP-Flow 最小可运行原型`

这表示：

- 仓库已经同时支持两条路径：
  - 原始 `diffusion`
  - 新增 `scpflow`
- `scpflow` 已经可以完成：
  - 训练
  - 验证
  - 测试
  - 生成图像导出
  - 预测结果导出
- 结构条件和缺访评估已经端到端接入
- 但整体仍然是原型阶段，距离论文级完整方案还有明显差距

## 三、原始 tHPM-LDM 关键模块定位

以下原始模块已经确认位置，并尽量复用：

1. `Autoencoder / E-D`
   - 文件：`train_vqldm_PMN.py`
   - 类：`VQModelInterface`
   - 方法：`instantiate_first_stage(...)`

2. `t-MSHF` + `PMQM`
   - 文件：`ldm/modules/condition_gen_MSTFCM_PopuMemory.py`
   - 当前 `scpflow` 通过 `ConditionGenMSTFCMPopuMemory` 复用

3. `noise estimator / denoiser`
   - 仍保留在原始 `diffusion` 路径中
   - 由 `DiffusionWrapper` 驱动

4. `latent alignment`
   - 原始 `diffusion` 路径仍按官方方式拼接历史 latent

5. `L_noise / L_assign`
   - 原始 `diffusion` 分支训练逻辑保持不变

6. 原始任务定义
   - 输入：历史 visits `1:T-1`
   - 输出：未来 visit `T`
   - SIGF clip 长度：`6`
   - 图像尺寸：`256 x 256`
   - 时间表示：相对首次随访的离散时间

## 四、已完成模块

### 1. Flow 主干替换

状态：`已完成（最小可运行版）`

已实现：

- 新文件：`models/flow_forecaster.py`
- 新类：`ProgressionFlowForecaster`
- 输入：
  - `z_t`
  - `tau`
  - `ch`
  - 可选 `cp`
  - 可选 `struct_cond`
  - 可选 `prev_latent`
- 输出：
  - `velocity`
  - `pred_target`
  - `pred_interval`
  - `pred_uncertainty`
  - `pred_structure`

说明：

- 当前是 flow 风格 latent 预测器
- 原始 diffusion 路径仍保留，便于对照

### 2. SCP-Flow 主训练入口

状态：`已完成`

已实现于：

- `train_vqldm_PMN.py`

新增内容：

- `--predictor_type diffusion|scpflow`
- 新 Lightning 模块：`SCPFlowLDM`
- 一组 SCP-Flow 专属参数：
  - bridge sigma
  - velocity loss weight
  - target latent loss weight
  - reconstruction loss weight
  - interval loss weight
  - uncertainty loss weight
  - consistency loss weight
  - structure loss weight

当前 `scpflow` 已支持：

- forward
- backward
- validation
- checkpoint 保存
- test 导出未来图像

### 3. 复用 tHPM-LDM 条件模块

状态：`已完成`

仍然复用的部分：

- 预训练 VQGAN 编码器/解码器
- 历史条件生成模块
- population memory 条件生成模块

当前 SCP-Flow 的条件包含：

- `ch`：来自 `ConditionGenMSTFCMPopuMemory`
- `cp`：来自 `ConditionGenMSTFCMPopuMemory`
- `struct_cond`：来自 dataloader
- `prev_latent`：作为病程锚点

### 4. 下一次随访间隔预测

状态：`已完成`

已实现：

- `ProgressionFlowForecaster` 中的 interval head
- `train_vqldm_PMN.py` 中的 interval loss
- `flow_predictions.csv` 导出
- `metrics/evaluate_scpflow.py` 中的 interval 评估

当前导出字段包括：

- `pred_interval`
- `target_interval`
- `abs_interval_error`

### 5. 不确定性预测

状态：`已完成（但质量仍弱）`

已实现：

- `ProgressionFlowForecaster` 中的 uncertainty head
- `train_vqldm_PMN.py` 中的不确定性损失
- `flow_predictions.csv` 导出
- `metrics/evaluate_scpflow.py` 中的不确定性统计

当前问题：

- 不确定性校准较弱
- 某些实验中，与真实误差的相关性仍可能为负

### 6. 病程一致性正则

状态：`已完成（最小版本）`

已实现于：

- `train_vqldm_PMN.py`

当前形式：

- 使用预测 latent 增量与上一段 latent 增量做约束
- 本质上是一个轻量的时间平滑 / 病程方向先验

当前局限：

- 仍然只是基础 latent consistency
- 还不是更强的多步、结构感知一致性正则

### 7. 结构条件接口

状态：`已完成（启发式代理版本）`

已接入文件：

- `datamodule/seq_fundus_2D_datamodule.py`
- `models/flow_forecaster.py`
- `train_vqldm_PMN.py`

当前结构输入包括：

- `disc_roi`
- `polar_roi`
- `vcdr`
- `ocod`

当前实现方式：

- 从每个 SIGF clip 中“最后一次可见随访图像”启发式生成
- `vCDR / OC-OD` 目前是代理值，不是人工标注

这意味着：

- 结构控制接口已经打通
- 但结构监督本身还不够临床可信

### 8. 缺访鲁棒性评估

状态：`已完成（已具备评估协议）`

已实现于：

- `datamodule/seq_fundus_2D_datamodule.py`
- `train_vqldm_PMN.py`
- `metrics/evaluate_scpflow.py`
- `metrics/run_scpflow_missing_visit_sweep.py`
- `metrics/aggregate_scpflow_missing_visit_results.py`

支持的缺访策略：

- `none`
- `tail`
- `uniform`
- `random`

当前控制参数：

- `--missing_visit_count`
- `--missing_visit_strategy`
- `--missing_seed`

现有结果目录：

- `results/scpflow_missing_tail_test`
- `results/scpflow_missing_sweep`

### 9. SCP-Flow 脚本入口

状态：`已完成`

已新增脚本：

- `scripts/train_scpflow.sh`
- `scripts/test_scpflow.sh`
- `scripts/metric_scpflow.sh`
- `scripts/train_scpflow.ps1`
- `scripts/test_scpflow.ps1`
- `scripts/metric_scpflow.ps1`

作用：

- 提供与官方 `tHPM-LDM` 脚本风格一致的 SCP-Flow 训练、测试、评估入口
- 同时兼容当前 Windows PowerShell 使用环境
- 用于后续正式实验，而不是继续依赖临时命令行

## 五、部分完成模块

### 1. 结构监督

状态：`部分完成`

已经做的：

- 结构特征已提取
- 结构嵌入已进入 flow 模型
- 已支持可选 `structure_loss`

还缺：

- 真实 `vCDR / OC-OD` 标签
- 更可信的 ROI 标注
- 更强、更稳定的结构监督方式

### 2. Polar 分支

状态：`部分完成`

已经做的：

- dataloader 已生成 `polar_roi`
- `StructureConditionEncoder` 已有单独 polar 编码支路

还缺：

- 专门的 polar 分支损失
- 单独 ablation
- 比简单加和更强的融合方式

### 3. 评测汇总

状态：`部分完成`

已经做的：

- 图像指标评估
- interval 指标评估
- uncertainty 指标评估
- missing-visit sweep 工具

还缺：

- 一张论文风格的统一结果表
- 与原始 diffusion 路径在同协议下的自动对比

## 六、未完成模块

### 1. 正式长训练版本

状态：`未完成`

当前仓库仍以 smoke 级别验证为主，尚未完成正式训练。

仍缺：

- 完整训练配方
- 稳定的最佳 checkpoint 选择方式
- 正式实验 checkpoint
- 完整长训练结果

### 2. 强不确定性建模

状态：`未完成`

仍缺：

- 更可信的不确定性建模
- 校准分析
- 误差与不确定性正相关的稳定结果

### 3. 强结构可控实验

状态：`未完成`

仍缺：

- 用户可控结构条件实验
- 结构条件扰动实验
- 证明结构变化确实能控制生成病程变化

### 4. 多步病程 rollout

状态：`未完成`

当前仅支持预测下一次 visit。

仍缺：

- 递推式 latent rollout
- 多步图像生成
- 多步 interval 预测
- 长时程一致性评估

### 5. SCP-Flow 与 tHPM-LDM 正式对照

状态：`未完成`

仍缺：

- 匹配训练预算
- 匹配测试预算
- 最终 ablation 表：
  - diffusion baseline
  - flow only
  - flow + interval
  - flow + uncertainty
  - flow + structure
  - flow + consistency

## 七、已验证证据

本地已经验证过：

1. `scpflow` 可完成单 batch forward/backward。
2. `scpflow` 可完成 smoke 训练并保存 checkpoint。
3. `scpflow` 可在 SIGF 上完成整套 test 推理并导出：
   - `*_gen.png`
   - `*_gt.png`
   - `*_rec.png`
   - `flow_predictions.csv`
4. 缺访设置会真实影响结果，尤其 `tail` 缺访对性能伤害明显。

## 八、当前推荐推进顺序

优先级顺序：

1. 训练一个非 smoke 的 SCP-Flow checkpoint，并开启 structure loss
2. 增加 `diffusion` 与 `scpflow` 的统一对比表
3. 改进 uncertainty calibration
4. 将结构监督从代理值升级为更可信的监督
5. 增加多步 rollout 实验

## 九、路线图

### Phase 1：稳定单步 SCP-Flow

目标：

- 让当前单步 SCP-Flow 足够稳定，可用于正式训练和比较

任务：

- 训练真实 checkpoint，而不是只停留在 smoke
- 确定稳定的验证集模型选择规则
- 同时监控图像质量、interval 误差和 uncertainty
- 验证 `structure_loss_weight` 是否真正有帮助

退出标准：

- 完整训练可跑通
- 最优 checkpoint 选择稳定
- 测试输出可重复

### Phase 2：结构控制升级

目标：

- 从“结构条件输入”升级为“结构可控”

任务：

- 改进 `disc_roi` 提取
- 提升 `polar_roi` 的使用方式
- 若有真实标签则替换代理 `vCDR / OC-OD`
- 增加 controllability 实验：
  - 改动结构条件
  - 观察生成病程变化

退出标准：

- 结构条件变化能产生可预测的输出变化
- 含结构条件模型优于无结构条件 ablation

### Phase 3：不确定性升级

目标：

- 让 uncertainty 真正可解释、可用

任务：

- 改进 uncertainty loss
- 做 calibration
- 比较 easy / hard / missing-visit 设置下的不确定性

退出标准：

- uncertainty 与误差稳定正相关
- 缺访条件下 uncertainty 会明显升高

### Phase 4：多步病程建模

目标：

- 将 SCP-Flow 扩展为多步未来预测

任务：

- 递推式 latent rollout
- 多步未来图像生成
- 多步随访间隔预测
- 长时程 drift / consistency 评估

退出标准：

- 多步预测完整跑通
- 有长时程结果表

### Phase 5：最终对照实验

目标：

- 判断 SCP-Flow 是否真正优于原始 diffusion 路径

任务：

- 匹配预算下比较：
  - 原始 `diffusion`
  - `scpflow`
  - `scpflow + structure`
  - `scpflow + uncertainty`
  - `scpflow + consistency`
- 输出图像指标、interval 指标、不确定性指标和缺访鲁棒性指标

退出标准：

- 一张最终 benchmark 表
- 一张 ablation 表
- 一张 missing-visit robustness 表

## 十、TODO 清单

### 高优先级

- [ ] 训练一个正式的 SCP-Flow checkpoint
- [ ] 加入 `diffusion` vs `scpflow` 的统一对比表
- [ ] 优化 checkpoint 选择规则，而不只看当前最小 `val/loss`
- [ ] 做 `structure_loss` ablation

### 中优先级

- [ ] 改进 uncertainty calibration
- [ ] 补齐 `count=2/3/...` 的 missing-visit sweep
- [ ] 汇总完整 missing-visit 曲线
- [ ] 在 validation 中正式记录结构指标
- [ ] 改进 ROI 提取质量

### 低优先级

- [ ] 增加多步 rollout
- [ ] 做结构可控实验
- [ ] 将 anatomy-aware consistency 与普通 latent consistency 区分开
- [ ] 把 SCP-Flow 实验封装成更完整的可复用脚本

## 十一、当前风险

1. 当前结构监督仍然依赖启发式代理标签，实验收益可能无法直接迁移到真实临床标注场景。
2. uncertainty head 目前还不够可信，可能增加复杂度但尚未带来稳定收益。
3. 目前已经验证的 checkpoint 多为 smoke 级别，因此性能结论仍是阶段性的。
4. 缺访鲁棒性对缺哪一次 visit 非常敏感，不做完整 sweep 时容易误读结果。

## 十二、SCP-Flow 新增文件

- `models/flow_forecaster.py`
- `metrics/evaluate_scpflow.py`
- `metrics/run_scpflow_missing_visit_sweep.py`
- `metrics/aggregate_scpflow_missing_visit_results.py`
- `scripts/train_scpflow.sh`
- `scripts/test_scpflow.sh`
- `scripts/metric_scpflow.sh`
- `scripts/train_scpflow.ps1`
- `scripts/test_scpflow.ps1`
- `scripts/metric_scpflow.ps1`

## 十三、SCP-Flow 修改文件

- `train_vqldm_PMN.py`
- `datamodule/seq_fundus_2D_datamodule.py`

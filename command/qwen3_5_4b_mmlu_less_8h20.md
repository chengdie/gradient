# Qwen3.5-4B：MMLU 完整 LESS 八卡运行记录

记录日期：2026-10-08。

> 更新：完整流程在 Flan V2 梯度提取阶段失败后，改为**一次只运行一个阶段**。当前推荐执行[第 9 节的拆分命令](#9-独立阶段命令当前推荐)，不再直接重跑第 2 节的 `run --stage all`。已有 warmup 保留复用。

## 1. 本次运行的目标

在 8 张 H20 上，对服务器已有的 Qwen3.5-4B 进行一次 **MMLU 定向 LESS 数据选择实验**，比较本次微调前后的 MMLU 表现。

主流程：

**原模型评测 → 随机 5% 数据 warmup → 全候选池逐样本梯度 → MMLU 参考梯度 → 匹配选数 → 从原模型重新微调 → 微调后评测。**

这不是只训练下载好的现成子集，而是用 Qwen 自身的 warmup 轨迹重新选择数据。

本文描述当前项目脚本的配置与执行逻辑。服务器实际生效的参数应以启动时打印的命令、各阶段的 `*.run.json` 和训练日志为准；后续修改本地配置不会自动修改已经启动的服务器进程。

## 2. 服务器位置与运行命令

### 2.1 路径

- 项目：`/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_awardcheng/LESS-main`
- 模型：`/mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_awardcheng/qwen3.5-4b`
- 数据：项目内 `data/`
- 实验产物：项目内 `out/qwen3.5-experiments/`
- 统一配置：`less/scripts/qwen3_5_config.sh`
- Shell 入口：`less/scripts/qwen3_5_experiments.sh`
- Python 编排入口：`less/experiments.py`

模型直接读取服务器已有目录，不需要复制进项目的 `models/`。

### 2.2 完整命令

以下记录对话中修复 Transformers 架构识别问题后给出的重跑命令；新的运行标记使用 `h20-less-8gpu-v2`：

```bash
cd /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_awardcheng/LESS-main

PYTHON_BIN=/usr/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash less/scripts/qwen3_5_experiments.sh run \
  --tasks mmlu \
  --methods base less \
  --selection-seeds 3 \
  --training-seeds 0 \
  --train-nproc 8 \
  --gradient-nproc 8 \
  --eval-nproc 8 \
  --stage all \
  --run-tag h20-less-8gpu-v2 \
  --execute
```

说明：

- 修复环境前的示例运行标记为 `h20-less-8gpu`，修复后建议换成 `h20-less-8gpu-v2`，避免与失败阶段的非空目录冲突。
- 用户后续提供了评测进度和训练日志，但没有重新贴出完整启动命令，因此不能仅凭日志确认当前进程的 `run-tag`。实际值见对应的 `*.run.json`。
- `/usr/bin/python` 是错误日志中实际使用的解释器；若后来切换虚拟环境，应将 `PYTHON_BIN` 指向该环境的 Python。
- 不要在这个命令外再套一层 `torchrun`。编排器自身只启动一次，各阶段由它分别启动八个 worker。
- `--execute` 才会实际执行。去掉它只打印计划，不训练、不评测，也不生成梯度。
- 这里没有显式写出的超参来自统一配置文件和 Python 默认值。以下表格按没有额外环境变量覆盖的情况说明。

### 2.3 参数含义

| 参数 | 本次含义 |
|---|---|
| `--tasks mmlu` | 只做 MMLU，不自动跑 BBH 或 TyDiQA |
| `--methods base less` | 先评测原模型，再进行 Qwen 自身重新选数及最终训练 |
| `--selection-seeds 3` | warmup 随机子集的抽样 seed 为 3 |
| `--training-seeds 0` | 最终微调的训练 seed 为 0；warmup 的训练 seed 默认也为 0 |
| `--train-nproc 8` | warmup 和最终微调都使用 8 卡 DDP |
| `--gradient-nproc 8` | 逐样本梯度提取使用 8 个独立 GPU worker |
| `--eval-nproc 8` | 前后评测都使用 8 卡分片推理 |
| `--stage all` | 执行该实验的完整阶段，不是执行所有实验组合 |
| `--run-tag ...` | 参与实验指纹，区分不同运行 |

## 3. 各阶段执行内容与资源

**所有阶段按顺序执行，同一组 8 张 H20 反复使用，峰值不是 24 张卡。**

| 顺序 | 阶段 | GPU 数量 | 每卡处理方式 | 是否更新权重 |
|---|---|---:|---|---|
| 0 | 检查文件、计算数据指纹、组织命令 | 0 | CPU 编排 | 否 |
| 1 | 原模型 MMLU 评测 | 8 | 每卡独立模型，评测 batch 4 | 否 |
| 2 | 随机子集 warmup | 8 | DDP，每卡 batch 2、累积 1 | 更新 LoRA |
| 3 | 候选数据逐样本梯度提取 | 8 | 每卡独立模型，每次一条样本 | 否，只求梯度 |
| 4 | MMLU 参考样本梯度提取 | 8 | 每次一条参考样本，普通梯度 | 否，只求梯度 |
| 5 | 特征匹配、排序、导出子集 | 0 | CPU 张量计算及磁盘读写 | 否 |
| 6 | 选中子集的最终微调 | 8 | DDP，每卡 batch 2、累积 1 | 更新新的 LoRA |
| 7 | 微调后 MMLU 评测 | 8 | 每卡独立模型，评测 batch 4 | 否 |

阶段 3、4 会对每个 warmup checkpoint 重复执行。一个子进程作业退出后，编排器才启动下一个；模型会在不同作业间重新加载。

### 阶段 1：原模型评测

加载原始 Qwen checkpoint，不加载本次实验的 LoRA adapter。

- 使用 MMLU 每学科 5-shot 提示。
- 使用独立测试题；加载器排除与参考集重叠的题目。
- 题目按 rank 分成八份，不补齐重复题目；各卡内部按长度排序并批量推理。
- 默认每卡 batch 为 4，最大上下文为 8192 tokens。
- MMLU 通过 A/B/C/D 的条件概率打分，不需要生成 512 tokens 的回答；`max_new_tokens=512` 主要用于生成类评测任务。
- 对单 token 选项，共享一次 prompt 前向计算四个选项概率。
- rank0 校验预测题目无重无漏，合并结果，输出学科宏平均 accuracy，采用 0～100 分制。

这里的 `base` 是“本次实验微调前的模型”，并不表示这个 Qwen checkpoint 从未经过官方后训练。

### 阶段 2：Warmup 训练

从四个训练来源合并后的候选池随机抽取 5%，使用 seed3。

| 来源 | 数据文件 | 当前已核对的样本数 |
|---|---|---:|
| Flan V2 | `data/train/processed/flan_v2/flan_v2_data.jsonl` | 100,000 |
| CoT | `data/train/processed/cot/cot_data.jsonl` | 100,000 |
| Dolly | `data/train/processed/dolly/dolly_data.jsonl` | 15,011 |
| OpenAssistant | `data/train/processed/oasst1/oasst1_data.jsonl` | 55,668 |
| 合计 | 四源混合池 | 270,679 |

抽样条数为 `floor(270679 × 0.05) = 13533`，不是逐来源等量抽取。训练前还会过滤编码后没有有效回答标签的样本，因此实际训练条数以日志为准。

Warmup 默认训练 **4 epochs**，每个 epoch 保存一个 checkpoint，同时导出：

- LoRA adapter；
- `optimizer_named.pt`：按参数名保存的 Adam 一、二阶矩；
- `selection_meta.json`：实际 epoch、训练步数、平均学习率、训练卡数及有效 batch。

正常完成四轮后，预计有四个选数 checkpoint。具体编号来自实际训练步数，不使用 README 中的固定编号。

**为什么选数之前先训练？** LESS 需要适应指令数据后的梯度，以及 Adam 的历史状态。Warmup 用来建立选数信号，不是最终交付的微调模型。

### 阶段 3、4：八卡逐样本梯度

对每一个 warmup checkpoint，按以下顺序执行：

1. Flan V2 候选梯度；
2. CoT 候选梯度；
3. Dolly 候选梯度；
4. OpenAssistant 候选梯度；
5. MMLU 参考梯度。

四个来源之间目前是串行的，但**每个来源内部都使用全部八张卡**，不是每个来源分两张卡。

每张卡：

- 加载相同 checkpoint、LoRA 参数和对应 Adam 状态；
- 负责一个不重叠的连续行号区间；
- 每次一条样本，独立前向、反向，再投影；
- 清空梯度后处理下一条；
- 不执行 `optimizer.step()`，不更新模型；
- 不使用 DDP，不做跨卡梯度平均。

候选侧计算 **Adam 风格更新方向**，目标参考侧计算 **普通 SGD 梯度**。MMLU 参考集为 57 学科 × 5 条，共 285 条；参考答案用于定义目标能力，不使用测试答案选数。

所有卡使用相同参数排列、投影种子和维度。默认把 LoRA 梯度投影为 **8192 维 FP32 特征**，不是将完整高维梯度长期写盘。

每卡在 `rank0/`～`rank7/` 下写独立分片。所有 worker 成功后，rank0 按原始行号合并元数据并检查完整性；不会再复制一整套特征张量。若任何 worker 失败，不发布全局完成 `manifest.json`。

其他边界：

- 候选样本编码后无监督标签：保留行号、标记无效，在选数中排除。
- 参考样本超过梯度长度上限：直接报错，不能静默截掉参考答案。
- 默认梯度长度上限为 4096；这与训练长度 1024、评测上下文 8192 是三项不同设置。
- 分片张量保存校验和；选数读取时再检查张量形状、有限值和校验和。

#### 梯度计算量

若正常保存四个 checkpoint：

- 候选记录：`270679 × 4 = 1082716` 份投影特征；无有效标签的候选只保存无效占位，不进行有效梯度计算。
- 目标参考记录：`285 × 4 = 1140` 份投影特征。
- 梯度作业：`4 × (4 个候选来源 + 1 个参考集) = 20` 个八卡作业，依次运行。

全候选池逐样本反向与投影通常是主要计算成本。八卡减少每卡负责的样本数，但不保证恰好提速八倍，仍受样本长度分布、投影开销和共享存储影响。

### 阶段 5：CPU 匹配与选数

调用 `less/data_selection/qwen_select.py`，当前实现不使用 GPU 做相似度计算。

顺序为：

1. 读取候选与参考的投影特征；
2. 逐样本 L2 归一化后匹配；
3. 根据各 epoch 的实际平均学习率做跨 checkpoint 加权；
4. 每个子任务内对参考样本得分平均；
5. 在 MMLU 各学科之间取最大得分；
6. 四个来源合并，全局排序，选取 13,533 条。

本次 `less` 模式会重新选具体样本。现成的 MMLU 子集文件存在时只用于对齐选数条数预算，不将它的样本直接作为本次自选结果。

输出 `selected.jsonl`、`indices.json` 和 `manifest.json`。保存的 JSONL 按原来源顺序排列；`indices.json` 保留选中样本的排序及分数。

### 阶段 6：最终微调

**重新加载原始 Qwen，初始化新的 LoRA adapter，在本次选中的数据上训练。不是接着 warmup adapter 继续训练。**

- 默认 **1 epoch**，不是 warmup 的 4 epochs。
- 使用选中子集的全部数据，即训练阶段 `percentage=1.0`，不再二次抽取 5%。
- 8 卡 DDP，每卡 batch 2、累积 1。
- 编码后无有效标签的记录仍可能被过滤，实际条数见训练日志。
- 当前不做逐 epoch benchmark 选优；最终评测使用该阶段结束后保存的 adapter。

### 阶段 7：微调后评测

加载“原始 Qwen + 阶段 6 的新 adapter”，使用与阶段 1 相同的 MMLU 数据、few-shot、seed 和打分协议，再用八卡评测。

对比两份 `metrics.json` 的 `score`，判断本次微调后分数是否提高。训练 loss 下降不能替代这个比较。

## 4. 默认超参与资源说明

| 配置项 | 当前默认值 |
|---|---|
| 模型精度 | BF16 |
| 普通注意力后端 | SDPA |
| LoRA 模块 | 文本分支普通注意力与线性注意力投影；视觉塔不训练 |
| LoRA rank / alpha / dropout | 16 / 64 / 0.1 |
| 优化器 | `adamw_torch`，weight decay 为 0 |
| 学习率 | `1e-5`，线性调度，学习率预热比例 3% |
| Warmup / 最终微调轮数 | 4 / 1 |
| 每卡训练 batch / 累积 | 2 / 1 |
| 有效训练 batch | `8 × 2 × 1 = 16` |
| 训练梯度检查点 | 默认关闭，以减少重计算 |
| 训练 / 梯度 / 评测长度 | 1024 / 4096 / 8192 |
| 每卡评测 batch | 4 |
| 目标参考与测试协议 | MMLU 参考 285 条，最终 5-shot 评测 |
| 评测 seed / 投影 seed | 42 / 0 |
| Warmup 抽样 seed / warmup 训练 seed / 最终训练 seed | 3 / 0 / 0 |
| 分词进程 / 每个训练进程的数据加载 worker | 4 / 2 |
| W&B | 关闭 |

本次是针对 Qwen3.5-4B 的运行配方，不是原论文 Llama-2 实验的逐项超参复现。八卡相同有效 batch 也不保证与单卡训练位级一致，随机数、采样器和浮点归约顺序可能不同。

### 显存与存储

- 普通 DDP 和独立梯度 worker 都是**每卡一份完整模型**，不是将模型切成八份。八卡主要提高吞吐，不将单卡模型权重占用降为八分之一。
- 当前加载完整 Qwen3.5 模型包装，视觉分支虽然冻结且不处理图像，仍有相应模型参数占用。
- 梯度阶段还持有 Adam 状态、单条样本的反向激活和随机投影缓存，实际显存峰值必须在服务器观察。
- 仅全候选池四个 checkpoint 的 8192 维 FP32 特征约为 `270679 × 4 × 8192 × 4 bytes`，即 **35.5 GB（十进制）**。这不包括模型、优化器、checkpoint、元数据和评测产物。
- CPU 匹配阶段 GPU 会空闲，这是当前实现的预期行为。

## 5. 输出文件与完成标记

所有以下路径相对于项目内 `out/qwen3.5-experiments/`。目录中的 `<fingerprint>` 由实际配置计算，不应手动猜测。

| 路径模式 | 内容 |
|---|---|
| `runs/mmlu-base-<fingerprint>/eval/` | 微调前评测 |
| `warmup/s3-<fingerprint>/` | Warmup 最终 adapter、训练指标和每 epoch checkpoint |
| `warmup/.../checkpoint-<step>/optimizer_named.pt` | 按 LoRA 参数名导出的 Adam 状态 |
| `warmup/.../checkpoint-<step>/selection_meta.json` | 选数所需的 epoch、平均学习率等 |
| `grads/<fingerprint>/adam/<source>/rank<rank>/` | 候选梯度分片 |
| `grads/<fingerprint>/reference/mmlu/rank<rank>/` | 目标参考梯度分片 |
| `selected/mmlu-s3-default-<fingerprint>/` | 本次 LESS 选中数据与索引 |
| `runs/mmlu-less-default-s3-t0-<fingerprint>/model/` | 最终微调 adapter 与训练指标 |
| `runs/mmlu-less-default-s3-t0-<fingerprint>/eval/` | 微调后评测 |

评测目录中：

- `predictions.rank<rank>.jsonl`：每卡的题目预测；
- `predictions.jsonl`：校验并合并后的全量预测；
- `metrics.json`：评分结果，最后写入，含 `completed: true`；
- `execution` 字段记录实际评测卡数、batch 和分片策略。

编排器在阶段输出目录旁保存 `*.run.json`，其中记录真实子命令、实验配置及指纹。模型配置和数据版本变化会影响输出指纹。

汇总不是 `run` 自动执行的阶段。整轮结束后可手动运行：

```bash
bash less/scripts/qwen3_5_experiments.sh summarize
```

它汇总同一输出根目录下所有已有完成评测，不只限于当前 run-tag；不同参数、任务和协议会分组。结果为 `summary.json` 与 `summary.csv`。本次只有一组选数 seed 和训练 seed，不能据此报告多次独立实验的稳定性。

## 6. 如何根据日志判断运行到哪一步

| 日志特征 | 对应阶段 |
|---|---|
| `evaluation.eval.qwen`、`[rank ...] 400/1755 examples` | 前测或后测；看 `--model_path` 是原模型还是 adapter、输出路径是 `base` 还是 `less` |
| `less.train.train`、输出路径含 `/warmup/`、`--num_train_epochs 4` | Warmup 训练 |
| `--export_optimizer_state True` | 为后续梯度选数保存状态的 warmup 配置 |
| `less.data_selection.qwen_gradients`、`rows [begin, end)`、`no gradient synchronization` | 八卡逐样本梯度提取 |
| `less.data_selection.qwen_select` | CPU 匹配选数 |
| `less.train.train`、输入为新生成的 `selected.jsonl`、输出路径含 `/runs/.../model` | 最终微调 |

用户最近贴出的 `loss`、`grad_norm`、`learning_rate`、`epoch=0.1667` 是训练日志。若来自 `/warmup/`，表示 warmup 第一轮约完成六分之一；默认四轮的整体进度约为 4.2%，不是整个 LESS 流程完成了 16.67%。仅凭 loss 行本身不能确定是 warmup 还是最终训练，需要结合该阶段启动命令。

## 7. 本次已经观察到的情况与注意事项

### 已有日志能证明什么

- 最初服务器报 `Transformers does not recognize qwen3_5`，失败于原模型评测的配置加载，未进入训练。
- 后续日志显示八个 rank 都成功加载权重，并推进到每卡 `400/1754` 或 `400/1755`，证明原模型评测当时在八卡运行。
- 该份评测日志合计 14,037 道题；这是该次数据/协议下的数量，不保证更换数据版本后仍相同。
- 后续出现了训练 loss 日志；按照同一条完整命令的顺序，应已进入 warmup，但当前尚未提供最终前测分数、梯度完成标记、选数结果及最终后测分数，不能记为整轮实验已完成。

### 加速组件警告

`The fast path is not available ... Falling back to torch implementation` 表示 Qwen3.5 线性注意力的可选加速组件不可用，采用 PyTorch 后备实现。它不是退回 CPU，也不是模型加载失败，但会影响速度。

可选组件涉及 `flash-linear-attention` 和 `causal-conv1d`，需要与服务器环境兼容。本文只记录，不执行安装。不要在运行中的环境直接更换依赖。

### 重跑与已有目录

- 同配置且阶段完成标记有效时，编排器可以跳过该已完成阶段。
- 阶段失败后留下非空输出目录，会被拒绝覆盖。当前不支持从失败样本位置自动续算，也不自动执行 Trainer 的 checkpoint resume。
- 改 `--run-tag` 会生成新的一组实验指纹；这是新运行，不等同于只恢复失败阶段，也可能导致已完成的前序阶段重跑。
- 不要删除正在运行的输出目录，不要将一次失败的部分梯度或部分预测手动伪装成已完成结果。

## 8. 这个命令没有执行的实验

`--stage all` 只表示当前指定组合的所有阶段。本次组合为：

**MMLU × base/默认 LESS × selection seed3 × training seed0。**

它不自动运行：

- `random` 随机选数微调基线；
- `transfer` 下载好的现成 LESS 子集迁移微调；
- BBH、TyDiQA；
- seed6、seed9 或其他训练 seed；
- SGD、Sign、去归一化、投影维度、checkpoint 数量消融；
- 全量候选数据的最终微调对照；
- 图像、视频等多模态训练。

因此，本次最终能直接回答的是：**Qwen 自身重新选择出的约 5% 数据，经过当前微调配方后，MMLU 是否优于微调前。** 要进一步证明 LESS 优于随机选数，还需要相同训练和评测预算下的 Random 对照。

## 9. 独立阶段命令（当前推荐）

### 9.1 拆分原则与当前状态

新入口为 `single --only <阶段>`。**每条命令只做一件事，结束后退出，不自动运行前置或后续阶段。** 不使用 `run --stage all`，也不使用原来的 `run --stage train`：后者在 `less` 模式仍会串起 warmup、梯度、选数和训练。

目前已知的服务器状态：

- 已保存并被梯度命令引用的 warmup 目录：`out/qwen3.5-experiments/warmup/s3-70f1a200e1839b1e`。
- 失败位置：`checkpoint-816` 对 Flan V2 的 Adam 梯度提取。
- 只收到了外层 `exit status 1`，没有完整的首个 worker 异常，因此**拆分入口不等于该错误已经修复**。
- 新入口会保留该阶段完整控制台日志，方便找到真正异常。
- 新的阶段输出统一放到 `out/qwen3.5-steps/`，不会覆盖先前失败的 `grads/3ff70709d9bd591e/adam/flan_v2`。
- 旧的前测结果若已有 `completed: true`，以及已完成的 warmup，都不需要重跑。

| `--only` | 本次命令只运行 | GPU | 是否自动继续 |
|---|---|---:|---|
| `before` | 一个任务的原模型评测 | 8 | 否 |
| `warmup` | 一次完整 warmup 训练 | 8 | 否，不提取梯度 |
| `train-grad` | 一个 checkpoint、一个候选来源的梯度 | 8 | 否，不自动换来源或 checkpoint |
| `reference-grad` | 一个 checkpoint、一个任务的参考梯度 | 8 | 否 |
| `select` | 将指定 checkpoint 的已有梯度匹配选数 | 0 | 否，不补算缺失梯度 |
| `finetune` | 使用显式指定的 JSONL 微调 | 8 | 否，不选数、不评测 |
| `after` | 显式指定 adapter 的任务评测 | 8 | 否，不训练 |

这些入口本次仅编写与静态核对，没有代为安装依赖或启动服务器实验。需先将更新后的项目文件同步到服务器。

### 9.2 设置变量，只需在当前终端执行一次

```bash
cd /mnt/gemininjceph3/geminicephfs/mmsearch-luban-universal/group_semantic_doc/user_awardcheng/LESS-main

export PYTHON_BIN=/usr/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORK="$PWD/out/qwen3.5-steps"
export WARMUP="$PWD/out/qwen3.5-experiments/warmup/s3-70f1a200e1839b1e"
export CKPT="$WARMUP/checkpoint-816"
```

若已换 Python 虚拟环境，修改 `PYTHON_BIN`。重连服务器后重新设置这些变量。`WORK` 只影响新的独立阶段产物，不会修改 `WARMUP`。

### 9.3 当前应该先执行：仅重跑 checkpoint-816 的 Flan 梯度

确认没有上一轮残留 GPU 任务后，**先只执行这一条**：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only train-grad \
  --tasks mmlu \
  --checkpoint-path "$CKPT" \
  --source flan_v2 \
  --gradient-nproc 8 \
  --work-dir "$WORK" \
  --execute
```

它不会重做前测或 warmup，不会自动跑 CoT、Dolly、OpenAssistant，不会自动训练模型。

- 输出：`$WORK/grads/checkpoint-816/adam/flan_v2/`
- 完成标记：上述目录内的 `manifest.json`
- 完整日志：`$WORK/grads/checkpoint-816/adam/flan_v2.console.log`
- 参数记录：`$WORK/grads/checkpoint-816/adam/flan_v2.run.json`

如果再次失败，查看日志，不要再把 loss 文本粘贴到 Shell 执行：

```bash
tail -n 200 "$WORK/grads/checkpoint-816/adam/flan_v2.console.log"
```

日志会包含 worker 的具体 Traceback，不只是外层 `exit status 1`。

### 9.4 同一 checkpoint 的其他梯度：逐条运行

Flan 成功后，再依次手动执行。下面是四条独立命令，**不要整段一次粘贴执行**；每条完成后再决定是否开始下一条。

只算 CoT：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only train-grad --checkpoint-path "$CKPT" --source cot \
  --work-dir "$WORK" --execute
```

只算 Dolly：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only train-grad --checkpoint-path "$CKPT" --source dolly \
  --work-dir "$WORK" --execute
```

只算 OpenAssistant：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only train-grad --checkpoint-path "$CKPT" --source oasst1 \
  --work-dir "$WORK" --execute
```

只算 MMLU 参考梯度：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only reference-grad --tasks mmlu --checkpoint-path "$CKPT" \
  --work-dir "$WORK" --execute
```

参考侧自动使用 SGD 梯度，候选侧默认 Adam。两侧默认相同长度 4096、维度 8192、投影 seed0。不要只改某一来源或某一侧的维度/长度后混合选数。

### 9.5 更换 checkpoint，重复上述五项梯度

先查看实际 checkpoint 目录，不根据训练日志猜测文件一定存在：

```bash
find "$WARMUP" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
```

如果实际目录确实为 `checkpoint-816`、`checkpoint-1632`、`checkpoint-2448`、`checkpoint-3264`，完成第一个后，将 `CKPT` 改为下一个：

```bash
export CKPT="$WARMUP/checkpoint-1632"
```

然后再次**逐条**执行 Flan、CoT、Dolly、OpenAssistant、MMLU 参考梯度命令。对其余 checkpoint 同理。

每次只加载一个 checkpoint，完成一个来源或参考集后退出。默认四个 checkpoint 共需要 20 条独立梯度命令。不要把“一个 checkpoint 的梯度完成”当作完整四 checkpoint LESS 已完成。

### 9.6 仅选数：所有所需梯度完成后执行

以下四个路径必须与实际已有 checkpoint 一致，且每个 checkpoint 对应的四来源和参考梯度都在同一 `WORK` 中完成：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only select \
  --tasks mmlu \
  --checkpoint-paths \
    "$WARMUP/checkpoint-816" \
    "$WARMUP/checkpoint-1632" \
    "$WARMUP/checkpoint-2448" \
    "$WARMUP/checkpoint-3264" \
  --percentage 0.05 \
  --work-dir "$WORK" \
  --execute
```

该命令读取每个 checkpoint 保存的真实平均学习率，生成匹配配置后只执行 CPU 选数。缺任何梯度 manifest 就报错，不会自动补跑梯度或 warmup。

- 新子集：`$WORK/selected/mmlu/selected.jsonl`
- 匹配配置：`$WORK/selected/mmlu.spec.json`
- 完整日志：`$WORK/selected/mmlu.console.log`
- 默认是全候选池的 5%，当前数据下约 13,533 条。

如果有意仅使用一个 checkpoint，可以只传一个路径，但这属于单 checkpoint 变体，不是原来的默认四 checkpoint 配方。

### 9.7 仅最终微调：显式传入已选数据

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only finetune \
  --tasks mmlu \
  --selected-file "$WORK/selected/mmlu/selected.jsonl" \
  --selection-seeds 3 \
  --training-seeds 0 \
  --epochs 1 \
  --work-dir "$WORK" \
  --execute
```

只从原始 Qwen 初始化新的 LoRA，训练这个文件一次，不重跑 warmup、梯度、选数或评测。

- 模型：`$WORK/models/mmlu-s3-t0/`
- 日志：`$WORK/models/mmlu-s3-t0.console.log`

### 9.8 仅微调后评测

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only after \
  --tasks mmlu \
  --adapter-path "$WORK/models/mmlu-s3-t0" \
  --work-dir "$WORK" \
  --execute
```

只评测指定的 adapter，不自动训练。

- 指标：`$WORK/eval/mmlu-after/metrics.json`
- 日志：`$WORK/eval/mmlu-after.console.log`

### 9.9 原模型评测和 warmup 的独立命令（已完成则跳过）

仅原模型评测：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only before --tasks mmlu --work-dir "$WORK" --execute
```

输出为 `$WORK/eval/mmlu-before/metrics.json`。已有上一轮成功的前测结果时可直接用于同协议比较，不需要重复。

仅 warmup（当前不要因为梯度失败就执行）：

```bash
bash less/scripts/qwen3_5_experiments.sh single \
  --only warmup --selection-seeds 3 --warmup-epochs 4 \
  --work-dir "$WORK" --execute
```

这是新 warmup，输出为 `$WORK/warmup/s3/`，不会恢复旧 warmup。若确实重新训练，则后续 `WARMUP` 必须改成这个新目录，不能将不同轨迹的 checkpoint 混用。

### 9.10 日志、失败重跑和限制

- 去掉任一命令的 `--execute` 只检查输入并打印该阶段命令，不启动 GPU 工作。选数预览仍要求其前置梯度文件存在。
- 每个实际运行阶段独立保存 `*.console.log` 和 `*.run.json`。日志文件放在结果目录旁边，不会干扰结果目录的非空保护。
- 相同参数且完成标记有效时可以跳过已完成的同一阶段。
- 失败留下的非空目录仍不会自动覆盖。排查后，在确认该阶段已退出的前提下，手动归档**这个失败目录**再重跑；不要删除整个 `WORK`，也不要删除旧 `WARMUP`。
- `--stage-output` 可改当前阶段的输出位置；但自动组装选数配置按 `WORK/grads/<checkpoint>/...` 查找，所以正式梯度建议保持默认结构。
- `single` 的输出由 `--work-dir` 和 `--stage-output` 决定，不由 `--run-tag` 或旧 `--output-dir` 决定。
- 一份 `WORK` 只对应一套任务/seed/梯度配方；改变梯度算法、长度、维度或 warmup 轨迹时使用新的工作目录，防止混用。
- 独立入口不自动运行全局汇总。前后比较请直接查看本次对应的两个 `metrics.json`；旧 `summarize` 主要扫描 `run` 模式的 `runs/*/eval/` 布局，不会自动扫描 `WORK/eval/`。
- 日志保存不等于任务后台化；长任务建议在服务器已有的持久终端会话中运行，避免 SSH 断线终止前台进程。

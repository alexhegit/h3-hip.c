# --serve 覆盖全部生成路径（未实现）

> 状态：提案。日期：2026-11-03。不改协议 v1alpha，不改 daemon 行为。
> 实现前先读本文。协议正文仍以 [`DESIGN_DHS.md`](DESIGN_DHS.md) 为准。

`--serve` 可以覆盖现在命令行上的全部生成路径：dense、FastH3、TAEH3、VSA、Pika H3 Flex、token reduction、Sol-Attn、`--fbc`。它不是另一套生成器。worker 最后调用的是同一次 `h3_generate`。一块 GPU 一次仍然只做一条视频。

本文只记录这个结论和实现时必须守住的边界。

## 命令行和 daemon 的差别

`./h3` 做完一条视频就退出。这次设过的环境变量和打开过的 LoRA 文件跟着进程消失。

`--serve` 是一个一直开着的进程。任务排队进来，做完一单再做下一单。进程不退出，上一单留下的状态还在。所以每一单必须自己声明路径，并且在下一单开始前把本单改过的状态收回。

不传新字段的任务必须和今天一样：质量档决定的步数、层数、`--reuse`，采样器保持进程启动时的选择，不合并 LoRA。fox-s2 的 md5 gate 因此不动。

## 现在已经接到生成参数上的

`h3d_worker.c` 的 `build_h3_params` 只写入尺寸、帧数、种子、输出路径、步数、层数、`--reuse`，以及参考文件。质量档 `fox-s2`、`fox-fast`、`cinematic`、`custom` 展开成 dense 用的步数、层数和复用。这就是当前 daemon 能稳定跑的 dense 路径。

协议里的模式有 T2VA、I2VA、FL2VA、Ref2VA。Ref2VA 会换到单独加载的权重上下文。FL2VA 还有一个现成缺口：请求里的图片在 worker 中一律变成 `H3_REFERENCE_IMAGE`，没有写成 `--first-frame` / `--last-frame`。覆盖 FL2VA 时要先把「参考图」和「首尾帧」分开，再沿用 CLI 已有的读图规则。

## 还没接到 `h3_params` 上的

这些开关在命令行里已经存在，daemon 的生成参数里还没有：

- `--fasth3-lora`、`--taeh3`、`--vsa`
- `--pika-lora`、`--pika-ref-lora`
- `--token-reduction`、`--sol-attn`、`--fbc`

`options.extra_args` 不能代替这些字段。它只被拼进任务回显的命令行字符串，worker 不读取它。质量档默认还会带上 20 步、45 层、`--reuse 2`。Pika 和 FastH3 都拒绝 reuse 大于 1、也拒绝层数不是 50。就算以后把 `extra_args` 接进真正的参数，预设仍会先送进这个非法组合。

命令行会拒绝的组合，daemon 也拒绝，并返回已有的 422。至少包括：Pika 与 FastH3 不能叠在同一单；VSA 必须带 FastH3 适配器；Pika 的 FL2VA 适配器不能拿去跑 Ref2VA，反过来也一样；Pika、FastH3、VSA 不与 token reduction、Sol-Attn、`--fbc`、SSD streaming、core reuse 大于 1 叠用。

## 常驻进程里必须收回的状态

权重本身每次从 MiniMax-H3 检查点读出再合并。没有 LoRA 的下一单不会把上一单的 LoRA 写进基础模型。要防的是进程里记住的开关和已经打开的文件。

1. **GPU 采样开关。** Pika 路径在 `H3_GPU_SAMPLER` 和 `H3_CPU_SAMPLER` 都没设置时，会把 `H3_GPU_SAMPLER` 设为 1，而且不在结束时清掉。同一进程里的下一单，哪怕是 dense 或 FastH3，也会跟着走 GPU 采样。命令行看不出这个问题，因为进程已经退出。
2. **LoRA 文件只打开一次。** FastH3 / Pika 共用的打开逻辑在 `fasth3.open` 已经为真时直接返回，不再看新的路径。下一单完全不用 LoRA 时，合并会被跳过，基础模型仍是基础模型。下一单要另一份文件时（Pika 换成 FastH3，或 FL2VA 适配器换成 Ref2VA），进程里还拿着第一次打开的那份。
3. **其余环境变量同样随单设置、随单恢复。** 包括 `H3_PIKA`、`H3_FASTH3_LORA`、`H3_TAEH3`、`H3_VSA`、`H3_TOKEN_REDUCTION`、`H3_SOL_ATTN`、`H3_FBC`，以及本单改过的 `H3_GPU_SAMPLER` / `H3_CPU_SAMPLER`。恢复的是进入该单之前的值，不是无条件删掉用户在启动 daemon 时自己设的值。

## 实现时要守住的

- 新字段默认关闭。旧客户端不传这些字段，任务行为与今天一致。
- 每单只使用自己声明的路径。上一单的采样器、LoRA 和有损开关不能漏到下一单。
- LoRA 路径变化，或从「有 LoRA」变成「没有」，都要关掉已打开的适配器再按本单处理。
- 适配器路径若由任务传入，必须落在 daemon 允许的根目录内，和 media root 的约束同一类。服务端也可以用环境变量配置默认路径，任务只传开关。字段名实现时再定，本文不改 v1alpha。
- 仍是单卡互斥队列。多任务并行不在本文范围内，见 [`DESIGN_MULTI_TASK.md`](DESIGN_MULTI_TASK.md)。

## 同一轮讨论里明确不做的

这三件事不要塞进这次 daemon 工作。

- **分辨率分桶不做允许列表。** `--width` 和 `--height` 继续是任意合法尺寸。Pika 公布的 480p / 768p 表是官方验证过的画布，不是模型的硬限制。若以后要贴近 Ref2VA 参考图的短边规则（768 档以下 1024，768 档及以上 2048），只改参考图的编码尺寸，不限制输出分辨率。
- **不与 diffusers 做逐位噪声对齐。** 收益只是同一颗种子能对照官方 latent，不提高画面质量。h3 自己的种子已经稳定。改成全局随机数会让现有 MiniMax、FastH3 和 fox-s2 md5 的成片都变。只在 Pika 下更换生成器，原有路径不动，但已有 Pika 样片会变；BF16 因子和默认 INT8 矩阵乘仍会使最终像素和官方 BF16 不同。
- **关键帧中心裁剪是 CLI 行为，不是 daemon 再写一套。** 仅 `--pika-lora` 的首帧和末帧按画布宽高比在原分辨率居中裁切，再缩放到画布。不带 Pika 时，首帧仍拉伸，末帧仍 cover。daemon 覆盖 FL2VA 时调用这条规则。

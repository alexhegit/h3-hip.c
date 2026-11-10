# --serve 覆盖全部生成路径

> 状态：部分实现（v0.19.0）。提案日期 2026-11-03，2026-11-10 按 v0.19.0 更新。
> 不改协议 v1alpha。协议正文仍以 [`DESIGN_DHS.md`](DESIGN_DHS.md) 为准。

`--serve` 可以覆盖现在命令行上的全部生成路径：dense、FastH3、TAEH3、VSA、Pika H3 Flex、token reduction、Sol-Attn、`--fbc`。它不是另一套生成器。worker 最后调用的是同一次 `h3_generate`。一块 GPU 一次仍然只做一条视频。

本文记录这个结论、v0.19.0 已经做到哪里，以及实现时必须守住的边界。

## 命令行和 daemon 的差别

`./h3` 做完一条视频就退出。这次设过的环境变量和打开过的 LoRA 文件跟着进程消失。

`--serve` 是一个一直开着的进程。任务排队进来，做完一单再做下一单。进程不退出，上一单留下的状态还在。所以每一单必须自己声明路径，并且在下一单开始前把本单改过的状态收回。

不传新字段的任务必须和今天一样：质量档决定的步数、层数、`--reuse`，采样器保持进程启动时的选择，不合并 LoRA。fox-s2 的 md5 gate（`make serve-regression`）因此不动。

## v0.19.0 已经接到生成参数上的

- 尺寸、帧数、种子、步数、层数、`--reuse`，质量档 `fox-s2`、`fox-fast`、`cinematic`、`custom`。
- 模式：T2VA、I2VA、FL2VA 写成 `first_frame` / `last_frame`（`refs.first_frame` / `refs.last_frame`，或 `refs.images` 位置形式）；Ref2VA 的图片、视频、音频写成参考文件，并换到单独加载的权重上下文。
- `options.extra_args` 由 worker 真正读取（不再只是回显），白名单：`--token-reduction`、`--sol-attn`、`--fbc`、`--vsa`、`--core-reuse N`、`--render-width N`、`--render-height N`、`--fasth3-lora PATH`、`--taeh3 PATH`。其他参数 422。
- `options.fasth3`：用 daemon 自己配置的 FastH3 dense 适配器（`H3D_FASTH3_LORA`，缺省为 `tools/download_fasth3_lora.sh` 的下载位置），任务只传开关，不传服务器路径；`/v1/info` 的 `adapters.fasth3` 表示是否可用。与 `--fasth3-lora PATH` 二选一。输出与 CLI `--fasth3-lora` 逐字节相同。
- `options.token_reduction`，与 `--token-reduction` 等价。

在一个 daemon 里依次跑 FastH3、VSA（`vsa-datafree` 适配器）、FastH3、fox-s2、864×480 Sol-Attn、864×480 dense，六单的 md5 都与同参数的 CLI 相同（gfx1201）。

## 还没接上的

- **Pika H3 Flex**（`--pika-lora`、`--pika-ref-lora`）。接上时要处理下面「GPU 采样开关」一条。
- **`token_reduction_schedule`**：它是进程级环境变量，daemon 返回 422。
- **提交时拒绝非法组合。** 命令行会拒绝的组合，daemon 也应在提交时返回 422。现在一部分只在生成时失败：例如 `options.fasth3` 配默认质量档（45 层、`--reuse 2`），任务先 202，再以 `FastH3 LoRA requires --reuse 1` 失败。至少包括：FastH3 / Pika 要 50 层与 `--reuse 1`；VSA 必须带 FastH3 适配器；Pika 与 FastH3 不能叠在同一单；Pika 的 FL2VA 适配器不能拿去跑 Ref2VA，反过来也一样；Pika、FastH3、VSA 不与 token reduction、Sol-Attn、`--fbc`、SSD streaming、core reuse 大于 1 叠用。

## 常驻进程里必须收回的状态

权重本身每次从 MiniMax-H3 检查点读出再合并；prepared DiT 缓存的键里含 LoRA 路径、VSA、Sol-Attn、`--fbc` 等，所以缓存不会拿错模型。要防的是进程里记住的开关和已经打开的文件。

1. **LoRA 文件只打开一次。** v0.19.0 已修：适配器路径变化时关掉旧的再按本单打开（FastH3 之后跑 VSA 拿到的是 `vsa-datafree`，不再是第一次打开的那份）。不用 LoRA 的一单不做合并。
2. **`H3_SOL_ATTN`。** v0.19.0 已修：`--sol-attn` 设置的值在下一单不带 Sol-Attn 时清掉；用户启动 daemon 时自己导出的值保留。
3. **GPU 采样开关，未修，Pika 接上前要修。** Pika 路径在 `H3_GPU_SAMPLER` 和 `H3_CPU_SAMPLER` 都没设置时，会把 `H3_GPU_SAMPLER` 设为 1，而且不在结束时清掉。同一进程里的下一单，哪怕是 dense 或 FastH3，也会跟着走 GPU 采样。命令行看不出这个问题，因为进程已经退出。
4. **其余环境变量。** `H3_FBC`、`H3_FASTH3_LORA`、`H3_PIKA`、`H3_TAEH3`、`H3_VSA` 每次 `h3_generate` 都按本单重写。新增的进程级开关要么同样每单重写，要么像 `H3_SOL_ATTN` 一样只收回本单自己设的值，不能无条件删掉用户在启动 daemon 时设的值。

## 实现时要守住的

- 新字段默认关闭。旧客户端不传这些字段，任务行为与今天一致。
- 每单只使用自己声明的路径。上一单的采样器、LoRA 和有损开关不能漏到下一单。
- LoRA 路径变化，或从「有 LoRA」变成「没有」，都要关掉已打开的适配器再按本单处理。
- 适配器路径若由任务传入，必须是 daemon 能读到的普通文件；更推荐服务端用环境变量配置默认路径，任务只传开关（`options.fasth3` 就是这样做的）。Pika 按同样方式加 `options.pika`，不改 v1alpha。
- 仍是单卡互斥队列。多任务并行不在本文范围内，见 [`DESIGN_MULTI_TASK.md`](DESIGN_MULTI_TASK.md)。

## 同一轮讨论里明确不做的

这三件事不要塞进这次 daemon 工作。

- **分辨率分桶不做允许列表。** `--width` 和 `--height` 继续是任意合法尺寸。Pika 公布的 480p / 768p 表是官方验证过的画布，不是模型的硬限制。若以后要贴近 Ref2VA 参考图的短边规则（768 档以下 1024，768 档及以上 2048），只改参考图的编码尺寸，不限制输出分辨率。
- **不与 diffusers 做逐位噪声对齐。** 收益只是同一颗种子能对照官方 latent，不提高画面质量。h3 自己的种子已经稳定。改成全局随机数会让现有 MiniMax、FastH3 和 fox-s2 md5 的成片都变。只在 Pika 下更换生成器，原有路径不动，但已有 Pika 样片会变；BF16 因子和默认 INT8 矩阵乘仍会使最终像素和官方 BF16 不同。
- **关键帧中心裁剪是 CLI 行为，不是 daemon 再写一套。** 仅 `--pika-lora` 的首帧和末帧按画布宽高比在原分辨率居中裁切，再缩放到画布。不带 Pika 时，首帧仍拉伸，末帧仍 cover。daemon 覆盖 FL2VA 时调用这条规则。

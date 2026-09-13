# h3-hip.c × DeepSeek Harness 集成 — 设计与协议文档

> 版本：v1.0　日期：2026-09-12　状态：M0 设计冻结，待进入实现
>
> 本文档分两部分，修改规则不同：
> **Part I 设计与决策**（宪法层，§1–§9）——变更需重新评审；
> **Part II 协议 v1alpha**（条文层，§10–§18）——可按版本规则修订，
> 每次 breaking change 递增协议版本号并写入发布说明。
>
> 两部分的交叉点仅一处：§9.4 管"协议该改什么"的决策，Part II 管
> "具体怎么写"。

---

# Part I　设计与决策

## 1. 背景与目标

h3-hip.c 是 MiniMax-H3 的 AMD GPU（ROCm/HIP）推理引擎，目前只有 CLI
形态，易用性受限于"记住 50+ 个参数 + 命令行操作"。本设计通过两层
增量改造，让用户用自然语言（Web / 手机 / 飞书 / 微信）驱动视频生成：

- **h3-hip.c 增加常驻 daemon 模式**（`--serve`）：进程常驻、权重流式、
  结构化进度、任务队列、优雅取消
- **新增 dsh 插件 `dsh-h3`**：把 daemon 包装成 DeepSeek Harness 的
  agent 工具，配 skill 指导提示词写作

远程访问、手机入口、IM 渠道、通知等**全部复用 dsh 生态现成插件**，
本设计不为它们写一行代码。

## 2. 总体架构（同机部署）

```
                 远程用户（浏览器 / 手机微信·飞书）
                        │
        ┌───────────────┼───────────────────────┐
        │ SSH 隧道 │ 带鉴权反代 │ Tailscale/组网 │
        ▼               ▼                       ▼
┌─────────────────────────────────────────────────────┐
│ dsh（唯一网络面）+ 生态插件：dsh-web-ui / dsh-im /    │
│ modlens /（起步期不装其他体验插件）                   │
│  ┌───────────────┐   ┌───────────────────────────┐  │
│  │ agent + skill │──►│ dsh-h3 插件（独立仓库）     │  │
│  │ (h3 skill)    │   │ · 语义化工具 schema        │  │
│  └───────────────┘   │ · 执行器接口（daemon/CLI） │  │
│                      │ · SQLite 任务记录          │  │
│                      │ · 协议版本协商             │  │
└──────────────────────┼───────────┬───────────────┘
                       │ loopback HTTP（协议见 Part II）
                       ▼
┌─────────────────────────────────────────────────────┐
│ h3 daemon（h3-hip.c 树内，--serve，experimental）    │
│  · 进程常驻，权重流式加载（page cache）              │
│  · per-GPU worker 互斥队列（MVP 单卡）              │
│  · 结构化进度 / 优雅取消 / 看门狗 / 产物配额         │
│  · CLI 路径完全不变，fox-s2 md5 gate 不动           │
└──────────────────────┬──────────────────────────────┘
                       ▼
              h3 推理核心（共享代码，不 fork）
```

铁律：**网络攻击面只有 dsh 一层**。daemon 绑 loopback，信任边界=本机
文件系统；"谁有资格生成视频"由 dsh 的访问控制（隧道/反代/IM 白名单）
完全接管。

## 3. 仓库分工

### 3.1 h3-hip.c（本仓库）——执行层

| 项 | 内容 |
| --- | --- |
| 产物 | `h3 --serve` daemon 模式 + CLI 行为保持不变 |
| 技术栈 | C / HIP / ROCm / Makefile（不引入 Node） |
| 质量门槛 | 现有 md5 gate 全保留；新增 daemon 路径 fox-s2 与 CLI 路径产物一致性 gate；崩溃/配额演练（§18） |
| 版本节奏 | 克制，跟随推理正确性；daemon 随 v0.12 以 experimental 发布 |
| 上游姿态 | B+：树内 experimental，协议带版本号，写明晋升标准 |
| issue 面 | 推理引擎用户（ROCm、显存、gate 失败） |

**新增代码范围**：HTTP 最小实现（keep-alive + chunked + SSE）、
JSON 协议层、worker 队列、看门狗、配额清理、路径约束、结构化进度。
全部围绕既有模型加载/推理代码做外壳，**不 fork 推理逻辑**。

### 3.2 dsh-h3（新独立仓库）——接口层

| 项 | 内容 |
| --- | --- |
| 产物 | dsh 插件（工具 + skill + 执行器抽象） |
| 技术栈 | Node.js / TypeScript（Cordis 插件框架） |
| 质量门槛 | 测试套件须在 fake daemon 与真 daemon 上全部通过（对拍） |
| 版本节奏 | 独立快速发版；dsh preview API breaking 时一周内可发适配版 |
| 依赖声明 | `h3-hip.c ≥ v0.12-exp`（协议 v1alpha）；pin 测试过的 dsh 版本范围 |
| issue 面 | agent 使用者（"为什么没生成/怎么改提示词"） |

**范围**：四个工具（h3_generate / h3_status / h3_list / h3_cancel）、
skill 文档、SQLite 任务记录、协议版本协商、CLI 直连执行器（Plan B）、
fake daemon（开发期替身，可放本仓 tests/fixtures）。

### 3.3 为什么分仓

- 技术栈/CI/受众/版本节奏完全不同（见讨论记录）
- 插件对 dsh preview 变化需快速发版，不能被 h3-hip.c 的发版节奏绑架
- 插件或 dsh 生态衰亡时，h3-hip.c 仓库纯度与声誉不受波及
- 生态惯例：dsh 插件均为独立仓库（dsh-im、dsh-web-ui 等）

### 3.4 配合契约（唯一协调点）

```
┌──────────────┐   HTTP/JSON + SSE, loopback    ┌──────────────┐
│   dsh-h3     │ ◄──── 协议 v1alpha ──────────► │  h3 daemon   │
│  (接口层)     │   X-H3-Protocol 版本协商        │  (执行层)     │
└──────────────┘                                └──────────────┘
```

- **协调只靠协议版本**（`/v1/info` 的 `protocol` 字段），不靠仓库同步、
  不靠语义化版本对齐
- 插件依赖"协议 ≥ v1alpha"，遇到 426 响应时引导用户升级对应一侧
- 两侧文档互相链接：h3-hip.c README 设"生态集成"节指向 dsh-h3；
  dsh-h3 README 顶部声明所需 h3-hip.c 最低版本

## 4. 已确认决策（评审问答 Q1–Q11 汇总）

| # | 决策点 | 结论 |
| --- | --- | --- |
| Q1 | 权重许可审查 | **MiniMax 社区许可由 h3-hip.c 作者审查；出结论前 IM 渠道不上线**（硬前置） |
| Q2 | daemon 崩溃语义 | 在制任务全部 `failed`，不自动重试；agent 询问用户后重提；`JOB_LOST_AFTER_RESTART` |
| Q3 | 超时看门狗 | per-job 超时强杀，标记 `timeout`；默认三档 1200/3600/7200 秒，可配置 |
| Q4 | 产物配额 | 输出根目录默认 **20GB**，满后 oldest-first 清理，运行中任务豁免 |
| Q5 | Plan B | 插件抽象执行器接口，保留 CLI 直连实现（同步+轮询） |
| Q6 | 日志 | 文件日志；插件 `~/.dsh-h3/logs/`，daemon 独立目录；INFO 默认/DEBUG 开关；10MB×5 轮转 |
| Q7 | 配置 | 环境变量为主 + 可选配置文件覆盖 |
| Q8 | 开发并行 | 契约先行：fake daemon 与 daemon 真身按协议并行开发 |
| Q9 | 等待期插件 | 起步期都不装（MVP 最简） |
| Q10 | 起步外部依赖 | dsh-web-ui + dsh-im（飞书，受 Q1 门控）+ modlens；其余 backlog |
| Q11 | 产出顺序 | 先协议草案（已产出，即 Part II） |

## 5. 规格文档索引

| 文档 | 位置 | 状态 |
| --- | --- | --- |
| daemon 协议（端点/schema/SSE/错误码/配置/验收） | 本文档 Part II | ✅ 已产出 |
| 插件工具 schema + skill 大纲 | dsh-h3 仓 | ⏳ 待产出（设计冻结后下一步） |
| 部署运维手册（隧道/反代/Tailscale/IM 白名单/配额运维） | dsh-h3 仓 docs/ | ⏳ 待产出 |
| 许可审查结论 | h3-hip.c 仓 | ⏳ 待作者完成（Q1，阻塞 IM 上线） |

## 6. MVP 切割

**做**：daemon（submit/status/cancel/events + 单 GPU 队列 + 结构化进度

- 路径约束 + 看门狗 + 配额）｜插件四工具 + skill + SQLite + 双执行器
- 协议版本协商｜fake daemon 对拍｜一致性 gate｜飞书入口（受 Q1 门控）

**不做（backlog）**：微信渠道、夜间批处理（dsh-timer-scheduler）、记忆
插件、多卡 worker 池（结构已预留）、PWA 推送、跨机 HTTP 开放（含 token
鉴权与 /v1/media 上传）、whale-girl、通知插件、daemon 空闲预热、
断点续跑、任务优先级。

## 7. 风险与开放项

| 风险/开放项 | 级别 | 应对 |
| --- | --- | --- |
| MiniMax 社区许可限制 IM 分发（Q1） | 🔴 阻塞 | 作者审查；结论前 dsh-im 不上线 |
| dsh preview API 破坏式更新 | 🟡 | 插件独立快发版；pin 版本；插件做薄 |
| daemon 路径与 CLI 路径结果漂移 | 🟡 | md5 一致性 gate 进 CI，硬约束 |
| SSE 断线/代理兼容 | 🟢 | Last-Event-ID 续传 + 15s 心跳 |
| MP4 在 dsh 前端的内嵌播放能力未验证 | 🟢 | 降级：海报帧 + 下载链接（poster 已在协议中） |
| 几何组合白名单需按实测填写 | 🟢 | 实现期由作者提供，写入 daemon 校验表 |
| 生态插件质量参差 | 🟢 | 装前用 dsh-plugin-check 体检；起步清单已收敛到 3 个 |

## 8. 里程碑

1. **M0 设计冻结**（本文档）✅ 本次完成
2. **M1 双端并行**：fake daemon + 插件骨架 ↔ daemon 真身 P0 端点
3. **M2 对拍联调**：插件测试套件双端通过；fox-s2 一致性 gate 绿
4. **M3 体验闭环**：skill 调优 + 飞书入口（Q1 通过后）+ 部署文档
5. **M4 发布**：h3-hip.c v0.12-exp（含 --serve）＋ dsh-h3 v0.1.0；
   提交 h3-hip.c 收录至 MiniMax-AI/awesome-minimax-h3-integration

## 9. 生态位与外部参考（2026-09-10 更新）

参考：MiniMax-AI/awesome-minimax-h3-integration（社区维护的 H3
集成总索引）。据此更新四点：

### 9.1 差异化定位（必须写进各自 README）

官方生态已在数据中心卡上提供一等 H3 serving：SGLang Diffusion
（官方 cookbook，day-0 AMD MI355X/MI300X via ROCm + AITER）与
vLLM-Omni（v0.26.0 起官方 recipe，OpenAI 兼容 /v1/videos，含 ROCm）。
本项目**不与它们竞争通用 serving**，差异化收窄为：

- **Strix Halo 核显（gfx1151）**：官方栈不覆盖
- **MI210/MI250（gfx90a）**：不在 AITER 主攻名单
- **antirez 式极简引擎形态**：单二进制、~无依赖、CLI 语义可复现
  （md5 gate）、权重流式适应小内存机器

### 9.2 协议形态的理由

vLLM-Omni 已暴露 OpenAI 兼容 videos API。本设计仍用自定义协议
v1alpha，理由：agent 场景需要 SSE 细粒度进度（分阶段、cache_hint）、
取消、配额、poster 帧等语义，OpenAI videos API 未覆盖。执行器抽象
预留了未来增加"OpenAI 兼容执行器"的扩展位（对接 SGLang/vLLM-Omni
跑 MI300X 时不改插件上层）。

### 9.3 skill 设计的官方依据

H3 提示词为**固定三段式结构**，内联 `<Picture X>` / `<Video X>` /
`<Audio X>` 引用标签与 `<d>` 对话标注；官方发布 Base（FL2VA）与
Ref2VA 两份提示词写作指南（HF MiniMaxAI/MiniMax-H3 docs/）。
dsh-h3 的 SKILL.md 采用官方三段式为规范主体并链接官方指南。

可参考的现成产物：

- `T8mars/minimax-h3-prompt-skill-T8`：可安装 agent skill + 案例库
- `SlavaSexton/ComfyUI-Agent-Kit`：agent 驱动 H3 的对标（Claude Code
  等经 ComfyUI 调 H3，含独立 H3 skill），其工具粒度设计值得研究

### 9.4 协议修订项（已回填 Part II）

- `refs` 容量上限按官方规格校验：Ref2VA ≤ 9 图 + 3 视频（各 2–15s）
- 3 音频 = 12 文件；FL2VA 为 0/1/2 张关键帧图
- mode→权重映射：FL2VA 与 Ref2VA 是独立 DiT 变体
  （transformer/ vs transformer_ref/，各 61.73 GiB），daemon 配置
  需支持两套权重路径（H3D_MODEL_PATH_FL2VA / _REF2VA）
- 几何白名单以 h3-hip.c 实测分辨率为准（官方管线支持 2K/15s，
  不代表本引擎路径已验证），写入 daemon 校验表时逐项标注来源

---

# Part II　h3-daemon 协议 v1alpha

> 状态：草案（experimental）。契约在标记 stable 前不保证兼容；每次
> breaking change 递增协议版本号并写入发布说明。
>
> 本文档同时作为以下三方的共同规格：
> 1. h3-hip.c 上游 `--serve` daemon 的实现依据
> 2. fake daemon（开发期替身）的实现依据
> 3. dsh 插件（执行器接口）的实现依据

## 10. 设计基线（已确认决策）

| 项 | 结论 |
| --- | --- |
| 部署形态 | 与 dsh 同机；daemon 默认绑定 `127.0.0.1`，跨机开放仅为部署选项 |
| 权重模型 | 进程常驻，权重流式加载（依赖 page cache），不做权重常驻 |
| 并发模型 | 每 GPU 一个 worker，per-GPU 互斥队列；worker = GPU 索引 + 互斥锁 |
| 崩溃语义（Q2） | daemon 崩溃 → 在制任务全部 `failed`，不自动重试 |
| 看门狗（Q3） | per-job 超时强杀，标记 `timeout`；默认三档 1200/3600/7200 秒，可配置 |
| 产物配额（Q4） | 输出根目录默认上限 **20GB**，满后 oldest-first 清理，阈值可配置 |
| 配置（Q7） | 环境变量为主 + 可选配置文件覆盖 |
| 日志（Q6） | 文件日志，INFO 默认 / DEBUG 环境变量开关，10MB × 5 份轮转 |
| 上游姿态 | 树内 `--serve`，experimental，复用 fox-s2 md5 gate 加 daemon 一致性测试 |

## 11. 传输与通用约定

- HTTP/1.1（keep-alive + chunked），无外部 HTTP 库依赖，C 侧为最小实现
- 请求/响应体一律 `application/json; charset=utf-8`
- 所有端点位于 `/v1` 前缀下
- 进度推送走 SSE：`text/event-stream`，单向，只推不接收
- 客户端每个请求携带头 `X-H3-Protocol: v1alpha`；服务端不识别时返回
  `426 Upgrade Required`（见 §16）

### 11.1 通用字段约定

- `job_id`：`^[a-z0-9][a-z0-9-]{7,31}$`，daemon 生成（uuid 短形）
- 时间一律 Unix 秒（整数）或 ISO 8601，本文示例用整数秒
- 文件路径一律绝对路径；daemon 只接受/返回受约束根目录内的路径
  （引用文件必须在 media root 内，产物一律在 output root 内）
- 枚举值大小写敏感，本文档全大写

## 12. 协议版本协商

`GET /v1/info` 响应携带 `protocol` 字段。客户端启动时应先调用并校验：

- 客户端 `X-H3-Protocol` 与服务端一致 → 正常服务
- 不一致 → 服务端返回 426，body 中指明 `server_protocol` 与
  `supported_protocols`。客户端据此降级或报错给 agent，由 agent 向用户
  说明"h3-hip.c 需要升级/插件需要升级"

协商结果由插件缓存，不必每次请求重验。

## 13. 端点

### 13.1 `GET /v1/info` — 服务信息与就绪状态

响应 200：

```json
{
  "protocol": "v1alpha",
  "h3_version": "0.12.0-exp",
  "ready": true,
  "unready_reason": null,
  "model_path": "/models/MiniMax-H3",
  "output_root": "/var/lib/h3d/outputs",
  "media_root": "/var/lib/h3d/media",
  "quota": { "limit_bytes": 21474836480, "used_bytes": 3221225472 },
  "workers": [
    { "gpu_index": 0, "arch": "gfx1151", "state": "idle",
      "current_job": null, "queue_depth": 1 }
  ],
  "limits": {
    "max_queued_jobs_per_client": 8,
    "prompt_max_chars": 8000,
    "ref_image_max_bytes": 52428800,
    "ref_video_max_bytes": 524288000,
    "ref_audio_max_bytes": 104857600
  }
}
```

- `ready=false` 时 `unready_reason` 为人类可读原因（权重缺失、ROCm 未
  初始化等）。此时提交任务返回 503。
- 插件应在会话早期调用一次，把 workers/limits 告知 agent（skill 可引用）。

### 13.2 `POST /v1/jobs` — 提交任务

请求体（语义化字段，daemon 内部翻译成 CLI 参数）：

```json
{
  "prompt": "string, 必填",
  "mode": "T2VA | I2VA | FL2VA | Ref2VA, 默认 T2VA",
  "size": { "width": 512, "height": 512 },
  "duration": { "frames": 22 },
  "quality": {
    "preset": "fox-s2 | fox-fast | cinematic | custom, 默认 fox-fast",
    "steps": 20, "layers": 45, "reuse": 2
  },
  "seed": null,
  "refs": {
    "images": ["/var/lib/h3d/media/a.png"],
    "video": null,
    "audio": null
  },
  "options": {
    "finalize_partial_on_cancel": false,
    "write_preview_frames": false,
    "extra_args": ["--token-reduction"]
  },
  "idempotency_key": "plugin 生成的可选去重键"
}
```

字段规则：

- `duration` 二选一：`frames` 或 `seconds`（互斥，同给返回 400）
- `quality.preset=cinematic` 时默认 steps=20/layers=45/reuse=2、864×480；
  `custom` 时必须显式给 steps/layers/reuse；preset 与显式字段同给时，
  **显式字段覆盖 preset**
- `size`/`duration` 合法组合由 daemon 按 h3-hip.c 已知的几何约束校验
  （如长片仅验证过 864×480），不在白名单内返回 422 并附允许值列表
- `mode=Ref2VA` 时 `refs.images` 必填（≥1），容量上限按官方规格：
  **≤ 9 图 + 3 视频（各 2–15s）+ 3 音频 = 12 文件**；超限返回 400
- `mode=FL2VA` 时 `refs.images` 为 0/1/2 张关键帧（首帧、末帧、首+末），
  `refs.video`/`refs.audio` 不接受；T2VA/I2VA 作为 FL2VA 的
  0/1 图特例走同一路径
- 提示词中引用文件须用官方 `<Picture X>` / `<Video X>` / `<Audio X>`
  标签语法（见 skill 文档）；daemon 不解析提示词，仅做文件存在性校验
- `refs.*` 路径必须在 media root 内，否则 400
- `options.extra_args` 为逃生口，daemon 原样追加到 CLI 参数；含
  已废弃或危险参数时返回 422 并指明拒绝项（如 `--frames-dir`、
  `--output` 由 daemon 托管，拒绝透传）
- `idempotency_key` 有效期内（建议 24h）重复提交返回原 job，不重复入队

响应 202：

```json
{
  "job_id": "k3v9a2x1",
  "status": "queued",
  "queue_position": 1,
  "events_url": "/v1/jobs/k3v9a2x1/events",
  "submitted_at": 1757414400
}
```

### 13.3 `GET /v1/jobs/{job_id}` — 查询任务

响应 200：

```json
{
  "job_id": "k3v9a2x1",
  "status": "running",
  "phase": "dit",
  "progress": {
    "step": 12, "total_steps": 20,
    "elapsed_sec": 612, "eta_hint_sec": 420
  },
  "request": { "...": "提交时的规范化请求回显（含解析后的 CLI 参数）" },
  "created_at": 1757414400,
  "started_at": 1757414700,
  "finished_at": null,
  "result": null
}
```

终态时 `status ∈ {done, failed, cancelled, timeout}`，`result` 为：

```json
{
  "mp4_path": "/var/lib/h3d/outputs/k3v9a2x1.mp4",
  "poster_path": "/var/lib/h3d/outputs/k3v9a2x1.poster.png",
  "duration_sec": 0.92,
  "width": 512, "height": 512, "frames": 22,
  "seed": 1991,
  "cli": "./h3 -d /models/MiniMax-H3 -p ... （完整参数回显，用于复刻）",
  "ffprobe": { "codec": "h264", "has_audio": true, "...": "摘要字段" }
}
```

失败终态时 `result=null`，`error` 为 `{code, message, details}`（见 §16），
并附 `cli` 便于复现排查。`poster_path` 由 daemon 从解码帧抽首帧生成，
供 IM/UI 展示。

查询支持 `GET /v1/jobs?status=running&limit=50`。

### 13.4 `POST /v1/jobs/{job_id}/cancel` — 取消

- queued/preparing/running 状态 → 转为 `cancelling`，尽力优雅停止
- 若 `options.finalize_partial_on_cancel=true` 且已有解码帧，封成草稿
  MP4 放入 `result.mp4_path`（`result.partial=true`）
- 已终态 → 返回 409
- 响应 200：终态 job 对象（同 13.3）

### 13.5 `GET /v1/jobs/{job_id}/events` — SSE 进度流

事件序列示例：

```
event: status
data: {"status":"queued","queue_position":1}

event: status
data: {"status":"preparing","phase":"loading_weights","cache_hint":"warm"}

event: progress
data: {"phase":"dit","step":3,"total_steps":20,"elapsed_sec":55,"eta_hint_sec":310}

event: progress
data: {"phase":"vae_decode","message":"decoding frames"}

event: status
data: {"status":"finalizing"}

event: done
data: {"job_id":"k3v9a2x1","result":{...同13.3...}}
```

- 终态事件类型即状态名：`done` / `failed` / `cancelled` / `timeout`，
  携带完整 job 对象；此后服务端关闭流
- 断线重连：客户端带 `Last-Event-ID` 头续传，daemon 保留每 job 最近
  50 条事件 1 小时（MVP 允许直接从头重放当前状态）
- 心跳：每 15 秒 `event: ping`，防代理断链

## 14. 任务生命周期

```
queued → preparing → running → finalizing → done
   |          |          |           |
   |          |          |           └─ failed (含崩溃恢复后)
   |          |          └─ cancelled / timeout
   |          └─ failed（权重加载失败等）
   └─ failed（校验通过但入队失败）
```

- `preparing` 阶段 `phase=loading_weights`，进度事件带 `cache_hint`
  （`warm`/`cold`/`unknown`），供 agent 向用户解释等待原因
- 看门狗超时：强杀子执行体，`status=timeout`，`error.code=JOB_TIMEOUT`，
  已解码帧是否保留遵循 `finalize_partial_on_cancel` 同规则
- **崩溃语义**：daemon 重启后，内存中所有非终态 job 视为 `failed`；
  产物目录中残留的半成品由配额清理机制回收（§15）。重启后 job 查询
  对已遗忘的 job_id 返回 404 + `error.code=JOB_LOST_AFTER_RESTART`，
  插件应提示 agent 向用户说明"daemon 重启，任务丢失，需重提"

## 15. 配额与清理

- 每次任务 finalizing 前检查 output root 用量；超限先触发清理：
  **oldest-first 删除整个任务目录**（mp4 + poster + 预览帧 + 元数据），
  直至低于上限的 90%
- 正在运行任务的目录永不清理
- 清理动作写入 daemon 日志（INFO），并在 `/v1/info` 的 `quota` 中反映
- media root 不做自动清理，仅文档建议定期手动清理

## 16. 错误码

统一错误体：

```json
{ "error": { "code": "VALIDATION_ERROR", "message": "人类可读概述",
             "details": [{ "field": "duration", "reason": "frames 与 seconds 互斥" }] } }
```

| HTTP | code | 含义 |
| --- | --- | --- |
| 400 | VALIDATION_ERROR | 字段校验失败，details 逐项列出 |
| 400 | PATH_OUT_OF_ROOT | 引用文件不在 media root 内 |
| 400 | REFS_LIMIT_EXCEEDED | Ref2VA 超过 9 图/3 视频/3 音频上限 |
| 404 | JOB_NOT_FOUND | job_id 不存在 |
| 404 | JOB_LOST_AFTER_RESTART | daemon 重启后丢失 |
| 409 | ALREADY_TERMINAL | 对终态 job 再取消 |
| 422 | UNSUPPORTED_COMBINATION | 几何/模式/preset 组合不在白名单，details 附允许值 |
| 422 | ARG_REJECTED | extra_args 含被拒参数，details 指明 |
| 426 | PROTOCOL_MISMATCH | 协议版本不符，body 附 `server_protocol`/`supported_protocols` |
| 429 | QUEUE_FULL | 该客户端排队任务超限（见 limits） |
| 500 | INTERNAL | 未预期错误，message 不泄露内部路径之外的敏感信息 |
| 503 | NOT_READY | daemon 未就绪，body 附 `unready_reason` |

## 17. 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `H3D_BIND` | `127.0.0.1` | 监听地址；显式设 `0.0.0.0` 即打开跨机口子 |
| `H3D_PORT` | `8571` | 监听端口 |
| `H3D_MODEL_PATH_FL2VA` | （二选一必填） | FL2VA DiT 权重目录（transformer/） |
| `H3D_MODEL_PATH_REF2VA` | （二选一必填） | Ref2VA DiT 权重目录（transformer_ref/）；daemon 按 mode 选权重 |
| `H3D_MODEL_PATH` | （兼容项） | 同时配置两者时的简写；显式分项配置优先 |
| `H3D_OUTPUT_ROOT` | `~/.h3d/outputs` | 产物根目录 |
| `H3D_MEDIA_ROOT` | `~/.h3d/media` | 引用文件根目录 |
| `H3D_QUOTA_BYTES` | `21474836480` | 产物配额（20GB） |
| `H3D_TIMEOUT_*` | `1200/3600/7200` | 三档超时分档映射（daemon 按任务规模选档） |
| `H3D_LOG_LEVEL` | `INFO` | DEBUG 开关 |
| `H3D_LOG_DIR` | `~/.h3d/logs` | 日志目录，10MB×5 轮转 |
| `H3D_GPUS` | `0` | worker 池 GPU 索引列表（逗号分隔），为后续多卡预留 |
| `H3D_CONFIG_FILE` | （无） | 可选配置文件路径，优先级高于环境变量 |

## 18. 明确不做与验收

### 18.1 明确不做（MVP 非目标）

- 用户认证/多租户（信任边界 = 本机文件系统权限）
- 跨机文件上传（`/v1/media` 端点不进 MVP；跨机时由部署方把文件放进 media root）
- TLS（loopback 默认无此需求；跨机由反向代理负责）
- 断点续跑、任务优先级、抢占
- WebSocket（SSE 单向流已够）

### 18.2 验收

1. **一致性 gate**：daemon 路径跑 fox-s2，产物 md5 与 CLI 路径一致
   （并入现有 halo-regression；MI210/MI300X 可选扩展）
2. **fake daemon 对拍**：插件测试套件在 fake 与真 daemon 上全部通过
3. **崩溃演练**：kill -9 daemon 后重启，在制任务按 §14 语义呈现，
   队列恢复可用
4. **配额演练**：输出根灌满后，新任务触发 oldest-first 清理且不触碰
   运行中任务

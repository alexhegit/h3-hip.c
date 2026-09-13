# DHS 设计文档 — DeepSeek Harness H3 Plugin

> 版本: 0.1.0 | 状态: Draft | 日期: 2026-09-13

## 1. 目标

在 h3-hip.c 中构建模型任务服务（h3d），并以独立 repo 开发符合 DeepSeek Harness Cordis 插件标准的专用插件，让用户在 DeepSeek Harness 中直接调用 H3 视频生成能力。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                DeepSeek Harness (dsh)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Cordis Plugin System                 │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │         dsh-plugin-h3-hip (TypeScript)          │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │  │
│  │  │  │ H3 Tool  │  │ H3 Skill │  │H3 Session│  │  │  │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │  │  │
│  │  └───────┼──────────────┼──────────────┼────────┘  │  │
│  └──────────┼──────────────┼──────────────┼───────────┘  │
└─────────────┼──────────────┼──────────────┼──────────────┘
              │              │              │
              ▼              ▼              ▼
         ┌────────────────────────────────────┐
         │      h3d HTTP API (C/RUST)        │
         │   POST /v1/jobs → SSE → Download  │
         └────────────────────────────────────┘
                         │
                         ▼
         ┌────────────────────────────────────┐
         │        h3-hip.c GPU Backend        │
         │    (ROCm HIP, MI300X/MI210/Halo)   │
         └────────────────────────────────────┘
```

## 3. h3d 服务端 API（已完成）

### 3.1 端点列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/v1/info` | 服务器状态、GPU 信息、quota |
| POST | `/v1/jobs` | 提交生成任务 |
| GET | `/v1/jobs/{id}` | 查询任务状态 |
| POST | `/v1/jobs/{id}/cancel` | 取消任务 |
| GET | `/v1/jobs/{id}/events` | SSE 实时事件流 |
| GET | `/v1/jobs/{id}/download` | 下载 MP4 结果 |

### 3.2 任务提交请求体

```json
{
  "prompt": "A red fox walks through fresh snow.",
  "width": 512,
  "height": 512,
  "frames": 22,
  "steps": 20,
  "seed": 12345,
  "denoise_reuse": 1,
  "dit_layers": 50,
  "token_reduction": false
}
```

### 3.3 任务状态响应

```json
{
  "job_id": "abc123xyz",
  "status": "running",
  "phase": "denoise",
  "progress": {
    "step": 10,
    "total_steps": 20,
    "elapsed_sec": 45,
    "eta_hint_sec": 45
  },
  "result": null
}
```

### 3.4 SSE 事件类型

| 事件 | 数据 |
|------|------|
| `status` | `{"status": "queued\|preparing\|running\|finalizing\|done\|failed"}` |
| `progress` | `{"phase": "...", "step": N, "total_steps": M}` |
| `done` | `{"mp4_path": "...", "width": N, "height": M, "seed": X}` |
| `failed` | `{"error": {"code": "...", "message": "..."}}` |
| `ping` | `{}` (心跳) |

### 3.5 环境变量配置

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `H3D_BIND` | `127.0.0.1` | 监听地址 |
| `H3D_PORT` | `8571` | 监听端口 |
| `H3D_MODEL_PATH` | - | MiniMax-H3 模型目录 |
| `H3D_GPUS` | `0` | GPU 列表（逗号分隔） |
| `H3D_OUTPUT_ROOT` | `~/.h3d/outputs` | 输出目录 |
| `H3D_QUOTA_BYTES` | `20GB` | 存储配额 |

## 4. DeepSeek Harness 插件设计

### 4.1 插件结构

```
dsh-plugin-h3-hip/
├── package.json          # 声明 dsh.bundle
├── cordis.patch.yml      # Cordis 配置层
├── src/
│   ├── index.ts          # 插件入口 (apply)
│   ├── h3-tool.ts        # H3 生成工具
│   ├── h3-skill.ts       # H3 技能封装
│   ├── h3-client.ts      # h3d HTTP 客户端
│   └── types.ts          # 类型定义
└── README.md
```

### 4.2 插件入口 (index.ts)

```typescript
import type { Context } from '@deepseek-ai/cordis'

export const name = 'dsh-plugin-h3-hip'
export const inject = ['tools', 'skills']

export function apply(ctx: Context) {
  // 注册 H3 工具
  ctx.tools.register({
    name: 'h3_generate',
    description: 'Generate video using H3 model on AMD GPU',
    parameters: {
      prompt: { type: 'string', description: 'Text prompt for video generation' },
      width: { type: 'number', default: 512 },
      height: { type: 'number', default: 512 },
      frames: { type: 'number', default: 22 },
      steps: { type: 'number', default: 20 },
    },
    execute: async (params) => {
      // 调用 h3d API
    }
  })

  // 注册 H3 技能
  ctx.skills.register({
    name: 'h3-video-gen',
    description: 'Generate video from text prompt using H3',
    tools: ['h3_generate'],
  })
}
```

### 4.3 h3d 客户端 (h3-client.ts)

```typescript
interface H3ClientConfig {
  baseUrl: string  // e.g. 'http://127.0.0.1:8571'
  timeout?: number
}

interface H3Job {
  job_id: string
  status: 'queued' | 'preparing' | 'running' | 'finalizing' | 'done' | 'failed'
  phase: string
  progress: { step: number; total_steps: number; elapsed_sec: number }
  result?: { mp4_path: string; width: number; height: number; seed: number }
  error?: { code: string; message: string }
}

class H3Client {
  constructor(private config: H3ClientConfig) {}

  async submitJob(params: H3GenerateParams): Promise<H3Job>
  async getJobStatus(jobId: string): Promise<H3Job>
  async cancelJob(jobId: string): Promise<void>
  async downloadResult(jobId: string, outputPath: string): Promise<void>
  subscribeEvents(jobId: string): AsyncGenerator<H3Event>
}
```

### 4.4 cordis.patch.yml

```yaml
- insert:
    id: h3
    name: '@anthropic/dsh-plugin-h3-hip'
    config:
      h3:
        baseUrl: 'http://127.0.0.1:8571'
        defaultWidth: 512
        defaultHeight: 512
        defaultFrames: 22
```

## 5. 用户工作流

### 5.1 服务端启动

```bash
# 在 AMD GPU 机器上启动 h3d
./h3 -d /path/to/MiniMax-H3 --serve

# 或通过环境变量配置
H3D_PORT=8571 H3D_GPUS=0,1 ./h3 -d /path/to/MiniMax-H3 --serve
```

### 5.2 插件安装

```bash
# 安装插件到 DeepSeek Harness profile
dsh plugin --profile myprofile add @anthropic/dsh-plugin-h3-hip
```

### 5.3 在 DeepSeek Harness 中使用

```
用户: Generate a 5-second video of a fox walking through snow
Agent: [调用 h3_generate 工具]
       → POST /v1/jobs {"prompt": "...", "frames": 120, ...}
       → 等待 SSE 事件
       → 下载 MP4
       → 返回给用户
```

## 6. 待实现项

### 6.1 h3d 服务端（需要完善）

- [ ] 引用图片/视频/音频支持（`--ref-image` 等）
- [ ] 认证/鉴权（API key 或 mTLS）
- [ ] 优雅关闭（graceful shutdown）
- [ ] 指标监控（/metrics 端点）
- [ ] 多模型热切换

### 6.2 dsh-plugin-h3-hip（新 repo）

- [ ] 初始化 Cordis 插件项目
- [ ] 实现 H3Client HTTP 客户端
- [ ] 实现 h3_generate 工具
- [ ] 实现 h3-video-gen 技能
- [ ] 实现进度回调和状态显示
- [ ] 错误处理和重试逻辑
- [ ] 打包和发布（npm）

### 6.3 集成测试

- [ ] 端到端测试：dsh → h3d → GPU → MP4
- [ ] 多并发任务测试
- [ ] 取消任务测试
- [ ] 网络断开恢复测试

## 7. 非目标（Out of Scope）

- Web UI（DeepSeek Harness 已有）
- 模型训练/微调
- 非 AMD GPU 支持（CUDA 等）

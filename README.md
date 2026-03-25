# Hyperagent

基于 Rust 的**自进化智能体系统**，通过进化循环迭代改进 AI 智能体。支持两种运行模式：

1. **进化引擎** — 进化"解决任务的代码"（执行→评估→变异→选择）
2. **自动研究** — Karpathy 风格，进化"系统自身的代码"（假设→实验→反思→提交 GitHub）

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hyperagent System                           │
├──────────────────────────┬──────────────────────────────────────┤
│    Mode 1: 进化引擎       │    Mode 2: 自动研究（Karpathy 风格）  │
│                          │                                      │
│  Task → Execute → Eval   │  Read → Hypothesize → Apply          │
│    → Mutate → Select     │    → cargo test → Reflect             │
│    → Meta-Mutate → Loop  │    → git commit → git push → Loop     │
├──────────────────────────┴──────────────────────────────────────┤
│  Memory Layer: Archive (disk persistence) + Lineage (血统树)     │
│  Thermodynamics: Energy, Entropy, Dissipation, Phase Transitions│
├─────────────────────────────────────────────────────────────────┤
│  LLM Layer: rig-core (OpenAI compatible)                        │
│  Self-Modification: auto_research.rs can modify its own code    │
└─────────────────────────────────────────────────────────────────┘
```

## 项目结构

```
src/
├── lib.rs                      # 库入口，公开 API 导出
├── main.rs                     # 进化引擎二进制入口
├── auto_research.rs            # 统一自动研究循环（Karpathy + 结构化自改进 + Web 搜索）
├── self_evolution.rs           # 递归自改进引擎
├── codebase.rs                 # 全局代码理解 + 上下文注入
├── web.rs                      # rig Tool: web_search (DuckDuckGo) + web_fetch
├── tools.rs                    # rig Tool: codebase_grep/search/read/tree
├── agent/
│   ├── mod.rs                  # Agent、MutationStrategy
│   ├── executor.rs             # LLM 任务执行
│   ├── mutator.rs              # 基于失败记录的智能体变异
│   ├── meta_mutator.rs         # 元学习：进化变异策略
│   └── population.rs           # 多智能体种群
├── eval/
│   ├── evaluator.rs            # LLM / 规则 / 集成评估器
│   └── benchmark.rs            # 基准测试任务
├── llm/
│   ├── client.rs               # LLMClient trait + 实现
│   └── prompts.rs              # 提示词模板
├── memory/
│   ├── archive.rs              # 有界存档（磁盘持久化）
│   └── lineage.rs              # 进化血统树（磁盘持久化）
├── runtime/
│   ├── loop_.rs                # 核心进化循环
│   ├── state.rs                # RuntimeState + 磁盘持久化
│   ├── thermodynamics.rs       # 热力学框架
│   ├── selection.rs            # 6 种选择策略
│   ├── constraints.rs          # 硬/软约束系统
│   ├── population.rs           # 种群进化
│   ├── environment.rs          # 环境交互
│   ├── local_runtime.rs        # 本地运行时
│   └── multi_agent_loop.rs     # 多智能体循环
└── bin/
    ├── research.rs             # 自动研究入口（主入口）
    └── self_evolve.rs          # 结构化自改进入口
```

## 快速开始

### 前置条件

- Rust 1.75+
- OpenAI 兼容的 API（支持自定义 base_url）

### 配置

创建 `.env` 文件：

```bash
# LLM 配置（支持任何 OpenAI 兼容 API）
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1  # 可选，自定义端点
```

### 运行模式

#### 模式 1：进化引擎（进化任务代码）

```bash
# 默认运行 5 代进化
cargo run

# 运行 N 代
ITERATIONS=10 cargo run
```

#### 模式 2：自动研究（自我进化 + GitHub）

```bash
# 默认：自动改进 + commit + push 到 GitHub
cargo run --bin research

# 安全观察模式（不提交）
RESEARCH_DRY_RUN=true cargo run --bin research

# 只 commit 不 push
RESEARCH_AUTO_PUSH=false cargo run --bin research

# 严格模式（测试 100% 通过才接受）+ 自定义迭代数
RESEARCH_STRICT=true RESEARCH_ITERATIONS=10 cargo run --bin research

# 后台无限运行（适合过夜）
nohup bash -c 'RESEARCH_AUTO_PUSH=true RESEARCH_ITERATIONS=1000 cargo run --bin research' > research.log 2>&1 &
```

#### 模式 3：结构化自改进

```bash
# 安全模式（默认 dry_run）
cargo run --bin self_evolve

# 实际提交
SELF_EVOLVE_DRY_RUN=false cargo run --bin self_evolve
```

### 查看结果

```bash
# 实验日志
cat .hyperagent/experiments/research_log.md

# 进化存档
cat .hyperagent/data/archive.json

# 血统树
cat .hyperagent/data/lineage.json

# Git 提交历史
git log --oneline
```

## 核心特性

### 进化引擎
- **多分支进化** — 每代生成多个变体，多样性选择
- **热力学框架** — Prigogine 耗散结构理论：温度退火、熵产生、Deborah 数
- **6 种选择策略** — 轮盘赌、锦标赛、截断、Boltzmann、排名、多样性保留
- **元变异** — 变异策略本身随代际进化
- **磁盘持久化** — Archive 和 Lineage 自动存档到 `.hyperagent/data/`

### 自动研究（Karpathy 风格）
- **极简循环** — 提出假设 → cargo check/test → LLM 反思 → git commit/push
- **自我修改** — 可以修改自己的源代码（包括 `auto_research.rs` 自身）
- **安全回滚** — 编译失败或测试退化自动 `git checkout`
- **自动 GitHub 推送** — 每次成功改进自动 commit + push
- **实验日志** — Markdown 格式记录每次实验的假设、结果、反思
- **dry_run / strict 模式** — 安全观察或严格验证
- **Web 搜索** — 每次迭代自动搜索外部知识（DuckDuckGo），注入研究上下文

### rig Tool 系统
所有工具实现 rig-core `Tool` trait，可供 LLM Agent 直接调用：

**Web 工具** (`src/web.rs`)：
- `web_search` — DuckDuckGo 搜索（无需 API Key）
- `web_fetch` — 抓取 URL + 提取纯文本

**本地代码工具** (`src/tools.rs`)：
- `codebase_grep` — 正则搜索源文件（支持扩展名过滤 + 上下文行）
- `codebase_search` — glob 模式文件查找
- `codebase_read` — 读取文件（支持分页）
- `codebase_tree` — 目录树结构列出

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 提供商 |
| `LLM_MODEL` | `gpt-4o` | 模型名称 |
| `LLM_API_KEY` | - | API 密钥 |
| `LLM_BASE_URL` | OpenAI 默认 | 自定义端点 |
| `RESEARCH_AUTO_PUSH` | `true` | 自动推送到 GitHub |
| `RESEARCH_DRY_RUN` | `false` | 安全模式 |
| `RESEARCH_STRICT` | `false` | 严格模式 |
| `RESEARCH_ITERATIONS` | `5` | 研究迭代数 |
| `RESEARCH_WEB` | `true` | 启用 Web 搜索 |
| `ITERATIONS` | `5` | 进化引擎迭代数 |

## 自动研究循环详解

```
┌──────────────────────────────────────────────────────┐
│  while iteration < max_iterations:                    │
│                                                      │
│    1. READ      读取目标源文件                         │
│    2. BASELINE  cargo test 获取基线                    │
│    3. WEB       [可选] LLM 生成查询 → 搜索 → 抓取页面  │
│    4. HYPOTHESIZE LLM + Web 上下文提出改进假设         │
│    5. APPLY     写入修改后的代码                       │
│    6. COMPILE   cargo check                           │
│       ├── FAIL → git checkout 回滚 → LLM 反思 → LOG   │
│       └── OK ↓                                       │
│    7. TEST      cargo test                            │
│       ├── REGRESS → git checkout 回滚 → LLM 反思 → LOG│
│       └── OK ↓                                       │
│    8. REFLECT   LLM 反思实验结果                      │
│    9. LOG       写入 research_log.md                  │
│   10. COMMIT    git commit                            │
│   11. PUSH      git push origin HEAD                  │
│                                                      │
│  end while                                            │
└──────────────────────────────────────────────────────┘
```

## 依赖

| Crate | 用途 |
|-------|------|
| `rig-core` | LLM 提供商抽象 + Tool trait |
| `tokio` | 异步运行时 |
| `serde` / `serde_json` | 序列化 + 磁盘持久化 |
| `anyhow` | 错误处理 |
| `tracing` | 结构化日志 |
| `chrono` | 时间戳 |
| `uuid` | 唯一标识 |
| `dotenvy` | 环境变量 |
| `reqwest` | HTTP 客户端（Web 搜索/抓取） |
| `regex` | 正则表达式（代码搜索） |
| `thiserror` | 错误类型定义 |

## License

MIT

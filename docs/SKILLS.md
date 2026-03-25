# Hyperagent System Skills

> 给 AI Agent 的完整项目知识，用于理解、维护和复刻此项目。

## System Identity

**Name**: Hyperagent - Self-Evolving AI Agent System
**Language**: Rust (edition 2021, async/tokio)
**Core Idea**: 用进化算法改进 AI 智能体，用 Karpathy 风格循环让系统修改自己的代码。

---

## Two Running Modes

### Mode 1: 进化引擎 (Evolution Engine)

进化"解决任务的代码"。循环：Execute → Evaluate → Mutate → Meta-Mutate → Select

入口：`cargo run`（src/main.rs）

核心循环在 `src/runtime/loop_.rs`：
```
FOR generation = 1 TO max_generations:
    1. 从 Archive 选择多样性父代
    2. LLM 执行任务 → 得到输出
    3. LLM/规则 评估打分 (0-10)
    4. LLM 变异生成新 Agent
    5. 定期元变异（进化变异策略本身）
    6. 比较 fitness = score × (1 + novelty_weight × novelty)
    7. 存入 Archive + Lineage
    8. 磁盘持久化（.hyperagent/data/）
```

热力学框架 (`src/runtime/thermodynamics.rs`)：
- 温度退火：T(g) = T₀ × rate^g（T₀=1.5, rate=0.9）
- Metropolis 接受准则
- 熵产生率、Deborah 数、适应度景观

### Mode 2: 自动研究 (Auto Research) — Karpathy 风格

进化"系统自身的代码"。循环：假设 → 实验 → 反思 → Git 提交

入口：`cargo run --bin research`（src/bin/research.rs）
引擎：`src/auto_research.rs`

```
WHILE iteration < max:
    1. 读取目标源文件 (src/下)
    2. cargo test 获取基线测试数
    3. LLM 读代码 → 提出 ONE 具体改进假设
    4. 写入修改后的完整文件
    5. cargo check → 失败则 git checkout 回滚
    6. cargo test → 测试退化则 git checkout 回滚
    7. LLM 反思实验结果
    8. 写入 .hyperagent/experiments/research_log.md
    9. git commit（成功时）
   10. git push origin HEAD（auto_push 开启时）
```

**关键特性**：
- 可以修改 `auto_research.rs` 自身（递归自我修改）
- dry_run 模式：编译通过也回滚，只观察
- strict 模式：测试 100% 通过才接受
- 每次迭代自动 git commit + push 到 GitHub
- 默认目标文件轮换：thermodynamics.rs → loop_.rs → auto_research.rs

### Mode 3: 结构化自改进 (Self Evolution)

更严格的自我修改模式。入口：`cargo run --bin self_evolve`（src/bin/self_evolve.rs）
引擎：`src/self_evolution.rs`

与 Mode 2 的区别：要求全部测试通过才接受，默认 dry_run。

---

## Project Structure

```
hyperagent/
├── Cargo.toml                  # rig-core, tokio, serde, anyhow, tracing, chrono, uuid
├── .env                        # LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
├── .hyperagent/
│   ├── data/
│   │   ├── archive.json        # 进化存档（持久化）
│   │   └── lineage.json        # 血统树（持久化）
│   ├── experiments/
│   │   └── research_log.md     # 自动研究实验日志
│   ├── config/
│   │   └── environment.json    # 运行环境配置
│   └── sessions/               # 会话数据
├── src/
│   ├── lib.rs                  # 公开 API 导出
│   ├── main.rs                 # 进化引擎入口
│   ├── auto_research.rs        # [核心] 统一自动研究循环
│   ├── self_evolution.rs       # 结构化自改进引擎
│   ├── agent/
│   │   ├── mod.rs              # Agent { id, code, prompt, generation, fitness, novelty }
│   │   ├── executor.rs         # LLM 执行任务
│   │   ├── mutator.rs          # 基于失败记录变异
│   │   ├── meta_mutator.rs     # 元学习：进化变异策略
│   │   └── population.rs       # 多智能体种群
│   ├── eval/
│   │   ├── evaluator.rs        # LLM/Rule/Ensemble 评估 (0-10分)
│   │   └── benchmark.rs        # 基准测试
│   ├── llm/
│   │   ├── client.rs           # LLMClient trait + LLMClientImpl (OpenAI 兼容)
│   │   ├── mod.rs              # 导出
│   │   └── prompts.rs          # 执行/变异/元/评估提示词
│   ├── memory/
│   │   ├── archive.rs          # Archive: 有界存档 + save_to_file/load_from_file
│   │   ├── lineage.rs          # Lineage: 血统树 + 持久化
│   │   └── mod.rs              # Record 结构体
│   ├── runtime/
│   │   ├── loop_.rs            # EvolutionLoop: 核心进化循环
│   │   ├── state.rs            # RuntimeState + with_persistence() + save()
│   │   ├── thermodynamics.rs   # EnergyState, DissipationScale, FitnessLandscape
│   │   ├── selection.rs        # 6种选择策略
│   │   ├── constraints.rs      # HardConstraints, SoftConstraints, CodeMetrics
│   │   ├── population.rs       # PopulationEvolution
│   │   ├── environment.rs      # Environment, Session
│   │   ├── local_runtime.rs    # LocalRuntime
│   │   └── multi_agent_loop.rs # 多智能体循环
│   └── bin/
│       ├── research.rs         # [主入口] 自动研究
│       └── self_evolve.rs      # 结构化自改进
├── examples/
│   └── basic.rs                # 基础用法示例
└── docs/
    ├── AR.md                   # 原始设计规范
    ├── arc2.md                 # 热力学理论框架
    ├── arc2_part2.md           # 相变与诊断
    ├── QUICKSTART.md           # 快速开始
    ├── SUMMARY.md              # 系统完成总结
    └── SKILLS.md               # 本文档
```

---

## Key Data Structures

### Agent
```rust
pub struct Agent {
    pub id: String,
    pub code: String,        // 可执行代码
    pub prompt: String,      // 系统提示词
    pub generation: u32,
    pub fitness: f32,
    pub novelty: f32,        // Jaccard 相似度计算
}
```

### ResearchConfig (Auto Research)
```rust
pub struct ResearchConfig {
    pub project_root: PathBuf,
    pub target_files: Vec<String>,      // 轮换改进的目标文件
    pub max_iterations: u32,
    pub auto_push: bool,                // 默认 true
    pub dry_run: bool,                  // 默认 false
    pub strict: bool,                   // 默认 false
    pub push_interval: u32,             // 0 = 每次成功都 push
    pub experiment_log_dir: PathBuf,    // .hyperagent/experiments
}
```

### Experiment
```rust
pub enum ExperimentOutcome { Improved, Neutral, Regressed, Failed }

pub struct Experiment {
    pub iteration: u32,
    pub file: String,
    pub hypothesis: String,
    pub outcome: ExperimentOutcome,
    pub tests_before: (u32, u32),
    pub tests_after: (u32, u32),
    pub reflection: String,
    pub timestamp: String,
}
```

### RuntimeState (Evolution Engine)
```rust
pub struct RuntimeState {
    pub config: RuntimeConfig,
    pub current_generation: u32,
    pub best_score: f32,
    pub best_agent: Option<Agent>,
    pub archive: Archive,
    pub lineage: Lineage,
    pub persist_path: Option<PathBuf>,  // 磁盘持久化路径
}
// with_persistence(config, ".hyperagent/data") 从磁盘加载
// save() 序列化 archive + lineage 到 JSON
```

---

## LLM Integration

```rust
// src/llm/client.rs
pub trait LLMClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
}

pub struct LLMClientImpl { ... }
impl LLMClientImpl {
    pub fn from_env() -> Result<Self>;  // 读 .env 配置
    pub fn provider(&self) -> &str;
    pub fn model(&self) -> &str;
}
```

支持的 provider：任何 OpenAI 兼容 API（通过 `LLM_BASE_URL`）。

---

## Persistence Layer

Archive 和 Lineage 通过 serde_json 序列化到磁盘：

```rust
// src/memory/archive.rs
impl Archive {
    pub fn save_to_file(&self, path: &Path) -> Result<()>;
    pub fn load_from_file(path: &Path) -> Result<Self>;
}

// src/memory/lineage.rs
impl Lineage {
    pub fn save_to_file(&self, path: &Path) -> Result<()>;
    pub fn load_from_file(path: &Path) -> Result<Self>;
}

// src/runtime/state.rs
impl RuntimeState {
    pub fn with_persistence(config: RuntimeConfig, dir: &str) -> Result<Self>;
    pub fn save(&self) -> Result<()>;
}
```

存储位置：`.hyperagent/data/archive.json` + `.hyperagent/data/lineage.json`

---

## Commands Reference

```bash
# 构建
cargo build

# 测试
cargo test

# 进化引擎（默认 5 代）
cargo run

# 自动研究（默认 auto_push, 5 轮）
cargo run --bin research

# 自动研究 - 安全观察
RESEARCH_DRY_RUN=true cargo run --bin research

# 自动研究 - 后台无限运行
nohup bash -c 'RESEARCH_AUTO_PUSH=true RESEARCH_ITERATIONS=1000 cargo run --bin research' > research.log 2>&1 &

# 结构化自改进
cargo run --bin self_evolve

# 查看结果
tail -50 research.log
cat .hyperagent/experiments/research_log.md
git log --oneline
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider name |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `LLM_API_KEY` | - | API key |
| `LLM_BASE_URL` | OpenAI default | Custom endpoint |
| `RESEARCH_AUTO_PUSH` | `true` | Auto git push |
| `RESEARCH_DRY_RUN` | `false` | Safety mode |
| `RESEARCH_STRICT` | `false` | Strict test mode |
| `RESEARCH_ITERATIONS` | `5` | Research loop count |
| `ITERATIONS` | `5` | Evolution engine generations |
| `NO_PROXY` | `localhost,127.0.0.1` | Proxy bypass |

---

## Recreating This Project From Scratch

如果要从零复刻此项目，按以下顺序实现：

### Phase 1: 基础设施
1. `cargo init --name hyperagent`
2. 添加依赖：`rig-core`, `tokio`, `serde`, `anyhow`, `tracing`, `chrono`, `uuid`, `dotenvy`
3. `src/llm/client.rs` — `LLMClient` trait + `LLMClientImpl`（OpenAI 兼容）
4. `src/agent/mod.rs` — `Agent` 结构体
5. `src/memory/archive.rs` — `Archive`（有界存档 + JSON 持久化）
6. `src/memory/lineage.rs` — `Lineage`（血统树 + 持久化）

### Phase 2: 进化引擎
7. `src/agent/executor.rs` — LLM 执行任务
8. `src/agent/mutator.rs` — 基于失败记录变异
9. `src/agent/meta_mutator.rs` — 元变异
10. `src/eval/evaluator.rs` — 评分
11. `src/runtime/loop_.rs` — 进化循环
12. `src/runtime/state.rs` — 状态管理 + 持久化
13. `src/main.rs` — 入口

### Phase 3: 热力学框架
14. `src/runtime/thermodynamics.rs` — 温度退火、Metropolis、熵
15. `src/runtime/selection.rs` — 6 种选择策略
16. `src/runtime/constraints.rs` — 约束系统

### Phase 4: 自我改进
17. `src/auto_research.rs` — Karpathy 循环（核心）
18. `src/bin/research.rs` — 入口
19. `src/self_evolution.rs` — 结构化自改进
20. `src/bin/self_evolve.rs` — 入口

### 关键设计决策
- **LLMClient trait**：便于 mock 测试和切换 provider
- **磁盘持久化**：Archive/Lineage 用 serde_json，进化引擎跨 run 保持记忆
- **git checkout 回滚**：编译失败或测试退化时自动恢复，保证系统不会损坏
- **auto_research 可以修改自己**：`auto_research.rs` 本身在 target_files 列表中
- **Neutral 改进保留**：测试数不变但编译通过 = 保留（宽松模式），有利于渐进式改进

---

## Known Issues & Gotchas

1. **48/49 测试**：当前有 1 个已存在的测试失败，不影响自动研究（宽松模式下只看是否退化）
2. **Novelty 坍缩**：多代进化后 novelty 从 1.0 降到 0.0，需要重启机制
3. **LLM 响应解析**：auto_research 依赖 `HYPOTHESIS:` + `IMPROVED_CODE:` 格式，有时解析失败
4. **编译超时**：单次 cargo check 最长 120s，cargo test 最长 300s
5. **git push 冲突**：如果本地和远程 diverge，push 会失败（需要手动 git pull）

---

**End of Skills Document**

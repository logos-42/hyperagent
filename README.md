# Hyperagent

一个基于 Rust 的**自进化智能体系统**，通过"执行、评估、变异、元变异"的进化循环迭代改进 AI 智能体。利用 LLM（通过 `rig-core`）既执行任务又改进智能体本身——包括改进变异策略（元学习）。

## 架构

```
                    +-----------------+
                    |  EvolutionLoop   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+-----+ +-----+-----+ +------+------+
        | 执行器     | | 评估器     | | 变异器     |
        | (LLM 调用) | | (评分 0-10) | | (LLM 调用) |
        +-----------+ +------------+ +------------+
                             |
                    +--------+--------+
                    |  元变异器       |
                    | (进化变异       |
                    |  策略本身)      |
                    +--------+--------+
                             |
                    +--------+--------+
                    |  记忆层         |
                    |  存档 +         |
                    |  血统树         |
                    +-----------------+
```

每次进化迭代遵循以下阶段循环：

1. **执行** —— 通过 LLM 使用当前智能体运行任务
2. **评估** —— 对输出进行评分（正确性、效率、鲁棒性，0-10 分）
3. **存档** —— 存储结果；记录到血统树中
4. **变异** —— 利用历史失败作为反馈生成新的智能体变体
5. **元变异**（定期） —— 进化变异策略本身
6. **选择** —— 与存档中的最优解比较；保留更优的智能体

## 项目结构

```
src/
├── lib.rs                  # 库根，重新导出所有公开类型
├── main.rs                 # 二进制入口
├── agent/
│   ├── mod.rs              # Agent、MutationStrategy 结构体
│   ├── executor.rs         # 通过 LLM 执行任务
│   ├── mutator.rs          # 基于失败记录的智能体变异
│   └── meta_mutator.rs     # 元学习：进化变异策略
├── eval/
│   ├── mod.rs              # 重新导出
│   ├── evaluator.rs        # LLM / 规则 / 集成评估器
│   └── benchmark.rs        # 基准测试任务套件和报告
├── llm/
│   ├── mod.rs              # 重新导出
│   ├── client.rs           # LLMClient trait + RigClient（OpenAI 兼容）
│   └── prompts.rs          # 提示词模板（执行/变异/元/评估）
├── memory/
│   ├── mod.rs              # 重新导出、Record 结构体
│   ├── archive.rs          # 有界存档，支持压缩和 top-k
│   └── lineage.rs          # 进化血统树
└── runtime/
    ├── mod.rs              # 重新导出
    ├── state.rs            # RuntimeState、RuntimeConfig、RuntimePhase
    └── loop_.rs            # 核心 EvolutionLoop
```

## 核心特性

- **基于 trait 的抽象** —— `LLMClient` trait 便于 mock 和切换 LLM 提供商
- **连接池** —— `Arc` 共享 HTTP 客户端，配合 `Semaphore` 并发控制
- **有界记忆** —— 存档达到容量上限时自动压缩
- **元学习** —— 变异策略本身会随着代际进化而改进
- **三种评估策略** —— 基于 LLM、基于规则的启发式、集成评估（多评估器取平均）
- **完整可观测性** —— 通过 `tracing` 进行结构化日志记录，配合基于阶段的状态机

## 快速开始

### 前置条件

- Rust 1.75+
- OpenAI 兼容的 API Key

### 使用方法

```rust
use hyperagent::{EvolutionLoop, RuntimeConfig, LLMConfig, RigClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 配置 LLM（默认读取 OPENAI_API_KEY 环境变量）
    let llm_config = LLMConfig::default();

    // 配置进化参数
    let runtime_config = RuntimeConfig {
        max_generations: 100,
        population_size: 5,
        top_k_selection: 3,
        checkpoint_interval: 10,
        meta_mutation_interval: 20,
    };

    // 构建并运行
    let client = RigClient::new(&llm_config)?;
    let mut loop_ = EvolutionLoop::new(client, runtime_config);
    let state = loop_.run("编写一个高效解决 X 问题的函数").await?;

    println!("{}", state.summary());
    Ok(())
}
```

### 配置项

| 字段 | 默认值 | 说明 |
|---|---|---|
| `LLMConfig.model` | `gpt-4o` | 模型名称 |
| `LLMConfig.api_key` | `$OPENAI_API_KEY` | API 密钥 |
| `LLMConfig.base_url` | `None` | 自定义端点（OpenAI 兼容） |
| `LLMConfig.max_concurrent` | `8` | 最大并发 LLM 请求数 |
| `RuntimeConfig.max_generations` | `100` | 最大进化代数 |
| `RuntimeConfig.meta_mutation_interval` | `20` | 每 N 代执行一次元变异 |

### 运行二进制文件

```bash
export OPENAI_API_KEY=sk-...
cargo run
```

## 依赖

| Crate | 用途 |
|---|---|
| `rig-core` | LLM 提供商抽象（OpenAI 兼容） |
| `tokio` | 异步运行时 |
| `serde` / `serde_json` | 序列化 |
| `anyhow` | 错误处理 |
| `tracing` | 结构化日志 |
| `chrono` | 时间戳 |
| `uuid` | 唯一标识 |
| `async-trait` | 异步 trait 支持 |

## 许可证

MIT

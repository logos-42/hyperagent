# Hyperagent 自进化系统 - 快速启动指南

## 系统概述

Hyperagent 是一个基于**耗散结构理论**和**进化算法**的自进化 AI 智能体系统。

```
                    能量输入 (LLM Token)
                          ↓
    ┌─────────────────────────────────────┐
    │         EvolutionLoop               │
    │  ┌──────────┐  ┌──────────┐        │
    │  │ 执行器   │→ │ 评估器   │        │
    │  └──────────┘  └──────────┘        │
    │       ↓              ↓              │
    │  ┌──────────┐  ┌──────────┐        │
    │  │ 变异器   │← │ 选择器   │        │
    │  └──────────┘  └──────────┘        │
    │       ↑                             │
    │  ┌──────────┐                       │
    │  │ 元变异器 │ (进化策略本身)        │
    │  └──────────┘                       │
    └─────────────────────────────────────┘
                          ↓
                    熵耗散 (Archive 压缩)
```

## 快速启动

### 1. 基础配置

```bash
export OPENAI_API_KEY=sk-...
cd /Users/apple/Downloads/hyperagent
cargo run
```

### 2. 自定义进化参数

```rust
use hyperagent::{SelfEvolvingConfig, SelfEvolvingSystem};

// 高探索性配置（适合初期）
let config = SelfEvolvingConfig::high_exploration();

// 高开发性配置（适合后期优化）
let config = SelfEvolvingConfig::high_exploitation();

// 自定义配置
let config = SelfEvolvingConfig {
    runtime: RuntimeConfig {
        max_generations: 100,
        population_size: 20,
        top_k_selection: 10,
        checkpoint_interval: 10,
        meta_mutation_interval: 20,
    },
    initial_temperature: 0.8,    // 高探索性
    annealing_rate: 0.95,        // 每代降温 5%
    population_size: 20,
    elite_ratio: 0.2,            // 保留 top 20%
    mutation_rate: 0.7,          // 70% 变异率
    selection_pressure: 0.3,     // 中等选择压力
    ..Default::default()
};

let mut system = SelfEvolvingSystem::new(config);
system.run("你的任务描述").await?;
```

## 核心参数调优

### 温度参数（探索/开发平衡）

| 参数 | 推荐值 | 效果 |
|-----|-------|------|
| `initial_temperature` | 0.8 (探索) / 0.3 (开发) | 初始探索强度 |
| `annealing_rate` | 0.95 | 每代温度衰减 |
| `min_temperature` | 0.1 | 最低温度限制 |

### 种群参数

| 参数 | 推荐值 | 效果 |
|-----|-------|------|
| `population_size` | 10-20 | 太小易早熟，太大计算贵 |
| `elite_ratio` | 0.2 | 精英保留比例 |
| `mutation_rate` | 0.7 | 变异个体比例 |

### 耗散参数

| 参数 | 推荐值 | 效果 |
|-----|-------|------|
| `entropy_threshold` | 0.5 | 熵超过阈值则注入多样性 |
| `stagnation_threshold` | 5 | 停滞 5 代后触发多样性注入 |

## 诊断工具

### 1. 检查系统状态

```rust
let scale = config.dissipation_scale();
println!("松弛时间：{:.2} 代", scale.relaxation_time);
println!("扩散长度：{:.2}", scale.diffusion_length);
println!("德博拉数：{:.2}", scale.deborah_number);

if scale.near_critical(0.2) {
    println!("⚠️ 系统接近临界点！");
}
```

### 2. 监测进化健康

```rust
let stats = PopulationStats::calculate(&population);
println!("种群大小：{}", stats.size);
println!("平均适应度：{:.2}", stats.mean_fitness);
println!("适应度标准差：{:.2}", stats.std_fitness);
println!("多样性指数：{:.2}", stats.diversity);
```

### 3. 问题诊断

| 症状 | 诊断 | 解决方案 |
|-----|------|---------|
| `diversity < 0.1` | 多样性过低 | 提高温度，增加变异率 |
| `stagnation_counter > 5` | 进化停滞 | 注入多样性，改变进化方向 |
| `De ≈ 1` | 临界状态 | 准备相变，可能跃迁到更优状态 |
| `entropy_production < 0` | 熵减（异常） | 检查能量输入是否充足 |

## 进化方向切换

```rust
use hyperagent::{ConstraintSystem, EvolutionDirection};

let mut constraints = ConstraintSystem::default();

// 初期：高探索
constraints.set_direction(EvolutionDirection::Exploration);

// 中期：效率优先
constraints.set_direction(EvolutionDirection::Efficiency);

// 后期：鲁棒性优先
constraints.set_direction(EvolutionDirection::Robustness);
```

## 实验模板

### 温度扫描实验

```rust
let temperatures = vec![0.1, 0.3, 0.5, 0.7, 0.9];

for temp in temperatures {
    let config = SelfEvolvingConfig {
        initial_temperature: temp,
        ..Default::default()
    };
    
    let mut system = SelfEvolvingSystem::new(config);
    system.run(task).await?;
    
    // 记录结果
    println!("T={:.1}: best_fitness={:.2}", temp, system.best_fitness);
}
```

### 相变点定位

```rust
// 扫描温度，测量比热（适应度涨落）
for temp in 0.1..=0.9, step=0.05 {
    let fluctuations = run_multiple_trials(temp);
    let specific_heat = variance(fluctuations) / (temp * temp);
    
    // 峰值对应相变点
    if specific_heat > threshold {
        println!("相变点 T ≈ {:.2}", temp);
    }
}
```

## 推荐工作流程

```
1. 初始探索阶段 (T=0.8, μ=0.8)
   ↓
2. 退火优化阶段 (T 从 0.8→0.3)
   ↓
3. 局部精炼阶段 (T=0.1, μ=0.3)
   ↓
4. 多样性注入（如果停滞）
   ↓
5. 返回步骤 2
```

## 示例输出

```
🚀 Starting Self-Evolving System
   Initial Temperature: 0.8
   Population Size: 20
   Task: Write a Rust function...

   Dissipation Scale:
      - Relaxation Time: 2000.00 generations
      - Diffusion Length: 14.14
      - Deborah Number: 200.00

Generation 10: T = 0.480
Generation 20: T = 0.288
Generation 30: T = 0.173
⚠️  Evolution stagnated at generation 35
💉 Injecting diversity to escape local optimum
Generation 40: T = 0.104

✅ Evolution completed after 50 generations
```

## 下一步

1. 阅读 `docs/arc2.md` 了解理论基础
2. 阅读 `docs/arc2_part2.md` 了解相变和诊断
3. 运行 `cargo run --example self_evolving` 查看演示
4. 修改 `examples/self_evolving.rs` 开始实验

## 参考

- 普利高津《非平衡系统的自组织》
- Kauffman《秩序的起源》
- Holland《自然和人工系统中的适应》

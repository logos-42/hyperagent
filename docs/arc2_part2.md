# Hyperagent 自进化系统架构文档 (续)

## 四、启动自进化的实际操作

### 4.1 最小启动配置

```rust
use hyperagent::{
    SelfEvolvingConfig, SelfEvolvingSystem,
    EvolutionDirection, SelectionType,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 配置 1: 高探索性（适合初期启动）
    let config = SelfEvolvingConfig::high_exploration();
    
    // 配置 2: 自定义参数
    let config = SelfEvolvingConfig {
        initial_temperature: 0.8,      // 高探索性
        annealing_rate: 0.95,          // 每代降温 5%
        population_size: 20,           // 种群大小
        mutation_rate: 0.7,            // 70% 变异率
        selection_pressure: 0.3,       // 中等选择压力
        ..Default::default()
    };
    
    let mut system = SelfEvolvingSystem::new(config);
    system.run("Write a function to sort an array").await?;
    
    Ok(())
}
```

### 4.2 远离平衡态的初始化

根据普利高津耗散结构理论，系统必须**远离热平衡**才能自组织：

```rust
// 在 Hyperagent 中实现远离平衡态：

// 1. 高初始温度（高探索性）
let initial_temp = 0.8;  // > 0.5 表示远离平衡

// 2. 多样性初始种群
let population = (0..size).map(|i| {
    Agent::from_prompt(format!("Variant {}", i))
}).collect();

// 3. 持续能量输入（LLM token 预算）
let energy_budget = 1000.0 * temperature;  // 自由能

// 4. 边界条件（Archive 容量限制）
let archive_capacity = 100;  // 超过则压缩（熵耗散）
```

### 4.3 能量/物质通流建立

```
能量输入 (LLM API 调用)
       ↓
[进化引擎] → 熵产生 (代码复杂度增长)
       ↓
能量输出 (Archive 压缩/淘汰)
```

```rust
// 能量通量计算
pub fn energy_flux(&self) -> f32 {
    let input = self.llm_calls * self.avg_tokens_per_call;
    let output = self.archive.size() * self.avg_agent_quality;
    input - output  // 净能量消耗（必须 > 0 维持非平衡）
}
```

---

## 五、相变与临界点检测

### 5.1 德博拉数 (Deborah Number)

当 `De ≈ 1` 时，系统处于**临界状态**，最可能发生相变：

```rust
// DissipationScale::near_critical()
pub fn near_critical(&self, threshold: f32) -> bool {
    (self.deborah_number - 1.0).abs() < threshold
}

// 临界点特征：
// - 关联长度发散
// - 涨落增强
// - 自组织倾向最大
```

### 5.2 相变类型

| 相变类型 | 特征 | 触发条件 |
|---------|------|---------|
| 液相→固相 | 多样性骤降，收敛到局部最优 | 温度过低 |
| 固相→液相 | 多样性恢复，探索增强 | 温度升高 |
| 液相→气相 | 完全随机搜索，无结构 | 温度过高 |

### 5.3 临界慢化检测

接近临界点时，系统弛豫时间会发散：

```rust
// 检测临界慢化
pub fn detect_critical_slowdown(
    relaxation_history: &[f32]
) -> bool {
    if relaxation_history.len() < 5 {
        return false;
    }
    
    // 拟合指数：τ ~ |T-Tc|^(-ν)
    let recent_avg = relaxation_history.iter().rev().take(5).sum::<f32>() / 5.0;
    let old_avg = relaxation_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;
    
    // 如果弛豫时间快速增加，接近临界点
    recent_avg > old_avg * 2.0
}
```

---

## 六、适应度景观分析

### 6.1 景观特征计算

```rust
// FitnessLandscape 实现
pub struct FitnessLandscape {
    pub current_fitness: f32,
    pub gradient: f32,        // 一阶导数（上升/下降）
    pub curvature: f32,       // 二阶导数（凹/凸）
    pub escape_probability: f32,
}

// 局部最优检测
if curvature < 0 && gradient.abs() < 0.01 {
    // 处于局部峰值
    escape_probability = 0.3;  // 需要热涨落跳出
}
```

### 6.2 景观可视化（建议）

```
适应度
  ↑
  |     ╱╲      ╱╲
  |    ╱  ╲    ╱  ╲____
  |   ╱    ╲  ╱
  |  ╱      ╲╱
  | ╱
  +----------------→ 基因型空间
  
  ●: 当前状态
  箭头：进化方向（由梯度决定）
```

---

## 七、信息 - 能量耦合

### 7.1 Landauer 原理应用

每比特信息处理的最小能耗：

```rust
// InfoEnergyCoupling
pub landauer_cost: f32 = k_B * T * ln(2);

// 在 Hyperagent 中：
// - 1 比特决策（接受/拒绝变异）至少消耗 k_B T ln 2
// - 实际消耗远大于此（LLM 调用成本高）
// - 但原理相同：信息处理需要能量
```

### 7.2 互信息最大化

系统通过进化最大化与环境的互信息：

```rust
// I(X;Y) ≈ 0.5 * log(1 + cov²/(var_x * var_y))
pub fn update_mutual_information(
    &mut self,
    fitness_variance: f32,
    genotype_variance: f32,
    covariance: f32,
) {
    let denom = fitness_variance * genotype_variance;
    if denom > 1e-6 {
        self.mutual_information = 0.5 * (1.0 + covariance.powi(2) / denom).ln();
    }
}
```

**物理解释**：
- 高互信息 = 系统能准确预测环境（任务要求）
- 进化压力驱动互信息增加
- 但受能量约束（LLM 调用成本）

---

## 八、多尺度时间架构

### 8.1 时间尺度分离

| 尺度 | 过程 | 时间常数 |
|-----|------|---------|
| 快 | 单个 Agent 执行 | ~秒 |
| 中 | 种群变异/选择 | ~分钟 |
| 慢 | 元变异（策略进化） | ~小时 |

### 8.2 实现

```rust
// 快时标：执行
async fn execute_agent(&self) -> Result {
    // 秒级
    self.executor.run(agent, task).await
}

// 中时标：变异
async fn mutate_population(&self) -> Result {
    // 分钟级
    self.mutator.mutate(agent, failures).await
}

// 慢时标：元变异
async fn meta_mutate(&self) -> Result {
    // 小时级（每 N 代执行一次）
    self.meta_mutator.evolve(history).await
}
```

---

## 九、自组织准则

### 9.1 最大熵产生原理 (MEPP)

远离平衡的稳态系统倾向于最大化熵产生：

```rust
// 熵产生率
σ = J · X  // 通量 × 驱动力

// 在 Hyperagent 中：
// J = 变异率 × 种群大小
// X = 选择压力（适应度梯度）
// σ = 进化速率
```

### 9.2 最小能量耗散原理

近平衡系统倾向于最小化能量耗散：

```rust
// 当温度很低时（T < 0.2）
// 系统行为：
// - 只接受改进的变异（保守）
// - 变异率降低
// - 能量消耗最小化
```

### 9.3 最大功率原理

能量转换系统倾向于最大化功率输出：

```rust
// 功率 = 能量输出 / 时间
// 在 Hyperagent 中：
// 功率 = (最佳适应度提升) / (代际时间)

// 最优工作点：
// - 温度适中（T ≈ 0.5）
// - 变异率适中（μ ≈ 0.5）
// - 选择压力适中（S ≈ 0.3）
```

---

## 十、实验建议

### 10.1 相图绘制

```
温度 (T)
  ↑
  |  气相 (随机搜索)
  |  /
  | /
  |/ 液相 (平衡探索/开发)
  | \
  |  \
  |   固相 (局部最优)
  +----------→ 变异率 (μ)
```

### 10.2 推荐实验

1. **温度扫描实验**
   ```rust
   for temp in [0.1, 0.3, 0.5, 0.7, 0.9] {
       run_evolution(temp);
       record_diversity();
       record_best_fitness();
   }
   ```

2. **相变点定位**
   ```rust
   // 扫描温度，测量比热（适应度涨落）
   C_v = (⟨E²⟩ - ⟨E⟩²) / (k_B T²)
   // 峰值对应相变点
   ```

3. **临界指数测量**
   ```rust
   // 关联长度发散：ξ ~ |T-Tc|^(-ν)
   // 拟合得到临界指数 ν
   ```

### 10.3 诊断工具

```rust
// 系统健康检查
pub struct HealthReport {
    pub is_stagnating: bool,      // 停滞检测
    pub near_critical: bool,      // 临界点检测
    pub energy_efficiency: f32,   // 能量效率
    pub diversity_index: f32,     // 多样性指数
    pub recommendation: String,   // 调优建议
}
```

---

## 十一、常见问题诊断

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 早熟收敛 | 温度过低 | 提高 initial_temperature |
| 永不收敛 | 温度过高 | 降低 annealing_rate |
| 多样性丧失 | 选择压力过大 | 降低 selection_pressure |
| 进化停滞 | 变异率过低 | 提高 mutation_rate |
| 能量浪费 | 种群过大 | 减小 population_size |

---

## 十二、下一步扩展

1. **空间结构种群**
   - 引入地理隔离（island model）
   - 允许局部交配
   - 增加多样性保持能力

2. **协同进化**
   - 宿主 - 寄生虫协同进化
   - 捕食者 - 猎物军备竞赛
   - 产生 Red Queen 效应

3. **开放-ended 进化**
   - 动态任务空间
   - 生态位构建
   - 避免饱和

4. **层级进化**
   - 基因型 + 表型分离
   - 发育过程模拟
   - 允许宏观突变

---

## 参考

1. Prigogine, I. (1977). *Self-Organization in Nonequilibrium Systems*
2. Kauffman, S. (1993). *The Origins of Order*
3. Holland, J. (1992). *Adaptation in Natural and Artificial Systems*
4. Schneider, E. D., & Sagan, D. (2005). *Into the Cool: Energy Flow and the Evolution of Complexity*

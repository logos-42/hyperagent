# Hyperagent Documentation Index

## Quick Navigation for AI Agents

### Essential Documents (Read in Order)

| Priority | Document | Purpose |
|----------|----------|---------|
| ⭐⭐⭐ | [SKILLS.md](SKILLS.md) | **Agent development guide - Start here** |
| ⭐⭐⭐ | [README.md](../README.md) | System overview and architecture |
| ⭐⭐ | [arc2.md](arc2.md) | Theoretical framework (thermodynamics) |
| ⭐⭐ | [QUICKSTART.md](QUICKSTART.md) | Practical implementation guide |
| ⭐ | [arc2_part2.md](arc2_part2.md) | Phase transitions and diagnostics |
| ⭐ | [SUMMARY.md](SUMMARY.md) | System completion summary |

---

## Key Files by Task

### Understand System Architecture
→ `README.md` → `src/runtime/loop_.rs` → `src/agent/mod.rs`

### Implement Evolution Loop
→ `SKILLS.md` (Evolution Loop Algorithm section) → `src/runtime/loop_.rs`

### Configure Parameters
→ `SKILLS.md` (Configuration Parameters section) → `examples/self_evolving.rs`

### Add New Features
→ `SKILLS.md` (Extension Points section) → Existing implementation

### Debug Issues
→ `SKILLS.md` (Debugging Guide) → `docs/QUICKSTART.md` (诊断工具)

### Understand Theory
→ `docs/arc2.md` → `docs/arc2_part2.md` → `src/runtime/thermodynamics.rs`

---

## Code Locations

| Component | File Path |
|-----------|-----------|
| Main Loop | `src/runtime/loop_.rs` |
| Agent | `src/agent/mod.rs` |
| Executor | `src/agent/executor.rs` |
| Evaluator | `src/eval/evaluator.rs` |
| Mutator | `src/agent/mutator.rs` |
| MetaMutator | `src/agent/meta_mutator.rs` |
| Selector | `src/runtime/selection.rs` |
| Thermodynamics | `src/runtime/thermodynamics.rs` |
| Constraints | `src/runtime/constraints.rs` |
| Archive | `src/memory/archive.rs` |

---

## Quick Commands

```bash
# Build
cargo build

# Test
cargo test

# Run
cargo run

# Self-evolving demo
cargo run --example self_evolving
```

---

## GitHub Repository

**URL**: https://github.com/logos-42/hyperagent

**Clone**: `git clone git@github.com:logos-42/hyperagent.git`

---

## Core Concepts (Cheat Sheet)

| Concept | Description |
|---------|-------------|
| **Evolution Loop** | Execute → Evaluate → Mutate → Meta-Mutate → Select |
| **Temperature** | Controls exploration vs exploitation (0.8 = explore, 0.1 = exploit) |
| **Dissipation** | Archive compression removes low-fitness agents |
| **Critical Point** | Deborah number ≈ 1 indicates phase transition |
| **Meta-Learning** | Mutation strategy itself evolves over generations |

---

## For AI Agents

**Goal**: Understand and replicate the Hyperagent system

**Steps**:
1. Read `SKILLS.md` for complete development guide
2. Study `src/runtime/loop_.rs` for core algorithm
3. Review `examples/self_evolving.rs` for usage patterns
4. Consult `docs/arc2.md` for theoretical foundation

**Key Skills Required**:
- Rust programming (async/await, traits, generics)
- LLM integration (via rig-core)
- Evolutionary algorithms basics
- Thermodynamics concepts (optional but helpful)

---

**Last Updated**: 2026-03-25
**Version**: v0.3.0

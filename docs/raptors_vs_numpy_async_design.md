# Raptors vs NumPy — Design & Value Proposition

## Overview

**Raptors** is a Rust-backed, NumPy-compatible numerical computing library designed specifically for **async-first Python applications** (e.g. FastAPI services).  
Unlike NumPy, Raptors provides **first-class async APIs** that enable safe, scalable CPU-bound computation without blocking the Python event loop.

This document explains **why Raptors exists**, **what problems it solves**, and **how it compares to NumPy** in real systems.

---

## Problem Statement

### NumPy in Modern Python Systems

NumPy was designed for:
- Scientific computing
- Interactive notebooks
- Batch workloads

Modern Python systems increasingly involve:
- Async web servers (FastAPI, Starlette)
- High-concurrency APIs
- CPU-heavy inference or scoring services

These worlds **do not align cleanly**.

### Core Issues with NumPy

1. **Blocks the async event loop**
2. **No awaitable API**
3. **Uncontrolled threading**
4. **No backpressure**
5. **Hard to integrate safely in async services**

Developers often resort to:
```python
await asyncio.to_thread(np.dot, a, b)
```
This works—but is fragile, verbose, and easy to misuse.

---

## Raptors Design Goals

Raptors is designed around these principles:

1. **Async-native API**
2. **Explicit CPU parallelism**
3. **Rust-managed execution**
4. **Safe FastAPI integration**
5. **Predictable performance**
6. **Zero-copy where possible**

Raptors does **not** attempt to replace NumPy everywhere—only where NumPy is a poor fit.

---

## Core Architecture

### High-Level Flow

Python (async)
↓
Raptors async API
↓ (GIL released)
Rust execution engine
↓
Thread pool (Rayon)
↓
CPU cores

### Key Decisions

- Rust handles **all CPU scheduling**
- Python handles **orchestration only**
- No Python threads for computation
- No multiprocessing overhead
- No silent oversubscription

---

## API Comparison

### NumPy

```python
result = np.dot(a, b)  # blocks
```

### Raptors

```python
result = await a.dot_async(b)  # non-blocking
```

Sync API also available:

```python
result = a.dot(b)
```

---

## FastAPI Example

### NumPy (Problematic)

```python
@app.post("/dot")
async def dot(a: list[float], b: list[float]):
    return np.dot(a, b)  # blocks server
```

### Raptors (Correct)

```python
@app.post("/dot")
async def dot(a: list[float], b: list[float]):
    return await rp.array(a).dot_async(rp.array(b))
```

### Benefits

- Server stays responsive
- Requests run concurrently
- CPU fully utilized
- No boilerplate threading logic

---

## Feature Comparison

| Feature | NumPy | Raptors |
|------|------|--------|
| Async-safe | ❌ | ✅ |
| Awaitable | ❌ | ✅ |
| Rust backend | ❌ | ✅ |
| GIL released | ⚠️ implicit | ✅ explicit |
| Backpressure | ❌ | ✅ |
| FastAPI-first | ❌ | ✅ |
| Predictable CPU usage | ❌ | ✅ |

---

## Performance Characteristics

### NumPy
- Implicit threading (BLAS)
- Hard to tune
- Can oversubscribe CPUs
- Event loop blocking

### Raptors
- Explicit thread pool
- Configurable limits
- Stable latency under load
- Designed for services

---

## Cancellation Semantics

- Python cancellation stops awaiting
- Rust kernel continues running
- Result is dropped
- No mid-kernel cancellation (CPU reality)

This matches industry-standard behavior (JAX, PyTorch, Polars).

---

## Use Cases Where Raptors Wins

✅ Async APIs  
✅ ML inference services  
✅ Vector similarity endpoints  
✅ Feature scoring  
✅ ETL microservices  
✅ CPU-heavy request handlers  

---

## Where NumPy Is Still Better

❌ Interactive notebooks  
❌ Exploratory analysis  
❌ Small scripts  
❌ Scientific research workflows  

Raptors is **not** a NumPy replacement—it is a **service-oriented compute engine**.

---

## Non-Goals

- Cooperative async math
- Step-by-step kernel yielding
- GPU execution (initially)
- Full NumPy API parity

---

## Summary

**Raptors fills a real gap**:

> NumPy is excellent at math.  
> FastAPI is excellent at concurrency.  
> Raptors makes them work together correctly.

It provides:
- Clean async APIs
- Predictable CPU usage
- Safe production defaults
- Modern Rust performance

---

## Final Verdict

If you are building:
- Async services
- CPU-bound APIs
- High-concurrency Python systems

**Raptors is strictly superior to NumPy**.

If you are doing:
- Research
- Exploration
- Offline analysis

**Stick with NumPy**.

Both tools can—and should—coexist.

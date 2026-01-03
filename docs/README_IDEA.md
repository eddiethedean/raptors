# Raptors 🦖
### NumPy-Compatible Compute Engine for Async Python Services

Raptors is a **Rust-powered, NumPy-compatible numerical computing engine** designed specifically for **async Python applications** such as FastAPI services.

> **Raptors is not “async math.”**  
> It provides **async job APIs** that allow CPU-bound numerical work to run **safely, predictably, and in parallel** without blocking the Python event loop.

---

## Why Raptors Exists

NumPy was designed for:
- Scientific computing
- Interactive notebooks
- Batch workloads

Modern Python systems increasingly involve:
- Async web servers (FastAPI, Starlette)
- High concurrency
- CPU-heavy request handlers

These worlds clash.

Developers today rely on:
```python
await asyncio.to_thread(np.dot, a, b)
```
This works—but it is fragile, verbose, and easy to misuse.

**Raptors makes this pattern first-class.**

---

## Key Features

- ✅ NumPy-compatible API (API-level)
- ⚙️ Rust backend with explicit CPU scheduling
- 🚫 Never blocks the async event loop
- 🔁 Async job submission (`*_async` methods)
- 🧠 Predictable performance under load
- 📦 Designed for FastAPI & async services

---

## Example: FastAPI

### Problematic (NumPy)

```python
@app.post("/dot")
async def dot(a: list[float], b: list[float]):
    return np.dot(a, b)  # blocks event loop
```

### Correct (Raptors)

```python
import raptors as rp

@app.post("/dot")
async def dot(a: list[float], b: list[float]):
    arr_a = rp.array(a)
    arr_b = rp.array(b)

    result = await arr_a.dot_async(arr_b)
    return {"result": result}
```

✔ Event loop stays responsive  
✔ CPU work runs in parallel  
✔ No thread boilerplate  

---

## Sync + Async APIs

```python
# Sync (science, scripts)
c = a @ b

# Async (services)
c = await a.matmul_async(b)
```

Async APIs are **explicit** and **opt-in**.

---

## What Raptors Is (and Isn’t)

### Raptors IS:
- A service-oriented compute engine
- A safe way to use NumPy-style math in async apps
- Built for production workloads

### Raptors IS NOT:
- A replacement for NumPy in notebooks
- A research framework
- Cooperative async math
- A SciPy clone

---

## How It Works

Python async code  
↓  
Raptors async API  
↓ *(GIL released)*  
Rust execution engine  
↓  
Thread pool (Rayon)  
↓  
CPU cores  

Python orchestrates.  
Rust computes.

---

## When to Use Raptors

✅ FastAPI / Starlette services  
✅ CPU-bound endpoints  
✅ Feature scoring APIs  
✅ Vector similarity services  
✅ ML inference (CPU-only)  

## When Not to Use Raptors

❌ Exploratory notebooks  
❌ Academic research  
❌ GPU-first workloads  

---

## Status

🚧 Early-stage / design-complete  
📖 Documentation-first  
🧪 Benchmarks coming soon  

---

## Philosophy

> **Async is for orchestration.  
> Rust is for computation.  
> Raptors connects them honestly.**

---

## License

Apache 2.0

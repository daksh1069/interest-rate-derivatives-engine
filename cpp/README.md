# C++ Monte Carlo Engine (Phase 4)

High-performance swaption pricer exposed to Python via **pybind11**, with
**OpenMP** path parallelism and **Sobol** quasi-random sequences.

## Planned sources
- `src/mc_engine.cpp` — exact Hull-White path generation, antithetic + control variates
- `src/swaption_pricer.cpp` — European pricer + Bermudan via Longstaff-Schwartz (LSM)
- `src/bindings.cpp` — `PYBIND11_MODULE(mc_engine, ...)`

## Targets to hit (from the plan)
- European MC vs Jamshidian analytical: match within 0.1 vol bp at 1M paths
- Bermudan ≥ European for all tested parameter sets
- 50–200× speedup vs pure-Python MC
- Sobol convergence faster than O(1/√N) standard MC

## Build
```bash
pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
The resulting `mc_engine*.so` imports directly from Python.

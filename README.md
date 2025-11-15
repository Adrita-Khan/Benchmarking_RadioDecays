# Radioactive Decay Chain Probability Estimation using Cython and Comparative Benchmarking

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **Note**: This project is currently in the preliminary stages of planning, implementation, and testing, and is subject to continuous advancements and modifications.

---

## Introduction

This project focuses on modeling radioactive decay chains through Monte Carlo simulations, with a primary goal of implementing the simulation in **Cython** to take advantage of its performance improvements over standard Python. We benchmark the Cython implementation against alternative Python libraries and explore implementations in other languages to identify optimal approaches for computational speed and memory efficiency.

### Objectives

- Model radioactive decay chains using Monte Carlo simulations
- Implement high-performance simulation in Cython
- Benchmark against multiple Python libraries and programming languages
- Optimize each implementation for maximum performance
- Collect and analyze performance metrics (execution time, memory usage)

---

## Background Theory

**Radioactive decay** is a stochastic process in which an unstable atomic nucleus loses energy through radiation emission. A **decay chain** refers to a sequence of successive radioactive decays that certain isotopes undergo until they stabilize.

### Key Concepts

| Concept | Symbol | Description |
|---------|--------|-------------|
| **Decay Constant** | λ | Rate at which a radioactive nucleus decays; probability per unit time of decay |
| **Half-Life** | T₁/₂ | Time required for half of the nuclei in a sample to decay |
| **Bateman Equation** | - | Analytical expression for time-dependent isotope behavior in decay chains |

### Bateman Equations

The Bateman equations provide the analytical solution for the number of nuclei in a radioactive decay chain as a function of time.

For a simple decay chain A → B → C:

**Isotope A (parent):**
```
N_A(t) = N_A(0) · e^(-λ₁t)
```

**Isotope B (daughter):**
```
N_B(t) = N_A(0) · (λ₁/(λ₂ - λ₁)) · (e^(-λ₁t) - e^(-λ₂t))     for λ₁ ≠ λ₂
```

**Isotope C (stable product):**
```
N_C(t) = N_A(0) · [1 + (λ₁/(λ₁ - λ₂))·e^(-λ₂t) + (λ₂/(λ₂ - λ₁))·e^(-λ₁t)]
```

Where:
- N_A(0) = initial number of nuclei of Isotope A
- λ₁ = decay constant of Isotope A
- λ₂ = decay constant of Isotope B
- t = time

These analytical solutions serve as validation benchmarks for our Monte Carlo simulations.

---

## Decay Chain Model

### System Description

We consider a simple three-component decay chain:

```
Isotope A → Isotope B → Isotope C (stable)
         λ₁           λ₂
```

| Component | Description | Decay Constant |
|-----------|-------------|----------------|
| **Isotope A** | Initial radioactive isotope | λ₁ |
| **Isotope B** | Intermediate radioactive product | λ₂ |
| **Isotope C** | Stable end product | - |

---

## Monte Carlo Simulation Algorithm

### Implementation Steps

| Step | Phase | Description |
|------|-------|-------------|
| **1** | **Initialization** | Set initial nuclei count N₀ for Isotope A; Define total time T and time step Δt |
| **2** | **Time Loop** | Iterate from t = 0 to T with increments of Δt |
| **3** | **Isotope A Decay** | For each nucleus: generate random r ∈ [0,1]; if r < λ₁·Δt, decay A → B |
| **4** | **Isotope B Decay** | For each nucleus: generate random r ∈ [0,1]; if r < λ₂·Δt, decay B → C |
| **5** | **Data Collection** | Record counts of nuclei A, B, and C at each time step |
| **6** | **Analysis** | Store time-series data for visualization and statistical analysis |

---

## Implementation Technologies

### Python Libraries & Extensions

| Technology | Type | Primary Use |
|------------|------|-------------|
| **Cython** | Python → C compiler | High-performance numerical computing |
| **Numba** | JIT compiler | Runtime optimization with decorators |
| **Pybind11** | C++ binding | Seamless C++ integration |
| **ctypes** | Foreign function library | C library interfacing |
| **Boost.Python** | C++ library | Advanced C++/Python binding |

### Additional Languages

| Language | Strengths |
|----------|-----------|
| **Julia** | High-performance scientific computing with Python-like syntax |
| **C++** | Maximum performance, low-level memory control |
| **MATLAB** | Built-in numerical algorithms, prototyping |
| **Mathematica** | Symbolic computation, mathematical analysis |
| **R** | Statistical computing, data visualization |

---

## References

### Documentation & Tools

| Resource | Link |
|----------|------|
| **Cython Documentation** | [https://cython.readthedocs.io](https://cython.readthedocs.io) |
| **Numba Documentation** | [http://numba.pydata.org/](http://numba.pydata.org/) |
| **Pybind11 Documentation** | [https://pybind11.readthedocs.io](https://pybind11.readthedocs.io) |
| **Boost.Python Documentation** | [https://www.boost.org/doc/libs/1_75_0/libs/python/doc/html/index.html](https://www.boost.org/doc/libs/1_75_0/libs/python/doc/html/index.html) |
| **Julia Language** | [https://julialang.org](https://julialang.org) |

### Theoretical Resources

| Topic | Link |
|-------|------|
| **Monte Carlo Methods** | [Wikipedia - Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method) |
| **Radioactive Decay Chains** | [Wikipedia - Decay Chain](https://en.wikipedia.org/wiki/Decay_chain) |
| **IAEA Notebook API** | [API v0 Notebook](https://notebooks.gesis.org/binder/jupyter/user/iaea-nds-lc_api_notebook-xa8lqngt/doc/tree/api_v0_notebook.ipynb) |

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Contact

**Adrita Khan**  
[Email](mailto:adrita.khan.official@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adrita-khan) | [Twitter](https://x.com/Adrita_)

---

*This project is intended for educational purposes and provides a foundation for further exploration into numerical simulations and performance optimization across different programming languages.*

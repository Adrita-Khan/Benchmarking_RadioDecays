# Radioactive Decay Chain Probability Estimation using Cython and Comparative Benchmarking
This project is currently in the preliminary stages of planning, implementation, and testing, and is subject to continuous advancements and modifications.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)  
## Introduction
This project focuses on modeling radioactive decay chains through Monte Carlo simulations, with a primary goal of implementing the simulation in **Cython** to take advantage of its performance improvements over standard Python. We will also benchmark the Cython implementation against alternative Python libraries (**Numba**, **Pybind11**, **ctypes**, **Boost.Python**) and explore implementations in other languages, including **Julia**, **C++**, **MATLAB**, **Mathematica**, and **R**.

The objective is to accurately estimate decay probabilities while optimizing for both computational speed and memory efficiency. This document provides a comprehensive, step-by-step guide, including source code and detailed explanations, designed for seamless integration on GitHub.


---

## Table of Contents
1. [Background Theory](#background-theory)
2. [Project Overview](#project-overview)
3. [Implementation Steps](#implementation-steps)
   - [Modeling the Decay Chain](#modeling-the-decay-chain)
   - [Monte Carlo Simulation](#monte-carlo-simulation)
   - [Cython Implementation](#cython-implementation)
4. [Implementations in Other Languages and Libraries](#implementations-in-other-languages-and-libraries)
   - [Numba](#numba)
   - [Pybind11](#pybind11)
   - [ctypes](#ctypes)
   - [Boost.Python](#boostpython)
   - [Julia](#julia)
   - [C++](#cpp)
   - [MATLAB](#matlab)
   - [Mathematica](#mathematica)
   - [R](#r)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)
8. [References](#references)

---


## Background Theory

**Radioactive decay** is a stochastic process in which an unstable atomic nucleus loses energy through radiation emission. A **decay chain** refers to a sequence of successive radioactive decays that certain isotopes undergo until they stabilize.

### Key Concepts

- **Decay Constant (λ)**: The rate at which a radioactive nucleus decays, representing the probability per unit time of a single nucleus decaying.
- **Half-Life (T₁/₂)**: The time required for half of the nuclei in a sample to undergo decay, providing a measure of the isotope's stability.
- **Bateman Equation**: An analytical expression that describes the time-dependent behavior of isotopes in a decay chain, allowing calculation of the number of nuclei present at any given time.


---

## Project Overview
### Objectives
- Model a radioactive decay chain using Monte Carlo simulations.
- Implement the simulation in Cython for performance optimization.
- Compare the performance against other implementations in various languages and libraries.
- Optimize each implementation for the best performance.
- Collect and analyze performance metrics like execution time and memory usage.

---

## Implementation Steps
### 1. Modeling the Decay Chain
We consider a simple decay chain:
- **Isotope A** decays to **Isotope B** with decay constant λ₁.
- **Isotope B** decays to **Isotope C** (stable) with decay constant λ₂.

### 2. Monte Carlo Simulation
#### Algorithm Outline
1. **Initialization**
   - Set initial number of nuclei **N₀** for Isotope A.
   - Set time parameters: total time **T**, time step **Δt**.

2. **Simulation Loop**
   - For each time step **t** from 0 to **T**:
     - For each nucleus of Isotope A:
       - Generate a random number **r** between 0 and 1.
       - If **r < λ₁ * Δt**, the nucleus decays (A → B).
     - For each nucleus of Isotope B:
       - Generate a random number **r**.
       - If **r < λ₂ * Δt**, the nucleus decays (B → C).
   - Record the number of nuclei of A, B, and C.

3. **Data Collection**
   - Store the counts at each time step for analysis.



## References
- **Cython Documentation**: [https://cython.readthedocs.io](https://cython.readthedocs.io)
- **Numba Documentation**: [http://numba.pydata.org/](http://numba.pydata.org/)
- **Pybind11 Documentation**: [https://pybind11.readthedocs.io](https://pybind11.readthedocs.io)
- **Boost.Python Documentation**: [https://www.boost.org/doc/libs/1_75_0/libs/python/doc/html/index.html](https://www.boost.org/doc/libs/1_75_0/libs/python/doc/html/index.html)
- **Julia Language**: [https://julialang.org](https://julialang.org)
- **Monte Carlo Methods**: [Wikipedia - Monte Carlo Method](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- **Radioactive Decay Chains**: [Wikipedia - Decay Chain](https://en.wikipedia.org/wiki/Decay_chain)
- **IAEA Notebook API**: [API v0 Notebook](https://notebooks.gesis.org/binder/jupyter/user/iaea-nds-lc_api_notebook-xa8lqngt/doc/tree/api_v0_notebook.ipynb)




---


<!-- **Full Source Code**: The full source code and detailed documentation for each implementation can be found on the [GitHub repository](https://github.com/yourusername/RadioactiveDecaySimulation). The repository includes instructions on how to set up and run each version of the simulation.

### GitHub Repository Structure
```text
RadioactiveDecaySimulation/
├── Cython/
│   ├── decay_chain.pyx
│   ├── setup.py
│   ├── main.py
├── Numba/
│   └── numba_decay_chain.py
├── Pybind11/
│   ├── decay_chain.cpp
│   └── setup.py
├── ctypes/
│   ├── decay_chain.c
│   ├── decay_chain.h
│   └── ctypes_interface.py
├── BoostPython/
│   ├── decay_chain.cpp
│   └── setup.py
├── Julia/
│   └── decay_chain.jl
├── C++/
│   └── decay_chain.cpp
├── MATLAB/
│   └── decay_chain.m
├── Mathematica/
│   └── decay_chain.nb
├── R/
│   └── decay_chain.R
├── Benchmarking/
│   ├── benchmark.py
│   └── results.csv
├── README.md
└── LICENSE
```

-- -->
---

## Contact

For any inquiries or feedback, please contact:

**Adrita Khan**  
[📧 Email](mailto:adrita.khan.official@gmail.com) | [🔗 LinkedIn](https://www.linkedin.com/in/adrita-khan) | [🐦 Twitter](https://x.com/Adrita_)



---

*This project is intended for educational purposes and provides a foundation for further exploration into numerical simulations and performance optimization across different programming languages.*

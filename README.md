# Radioactive Decay Chain Probability Estimation using Cython and Monte Carlo Simulations

This repository contains the implementation of a **Monte Carlo Simulation** to model the **radioactive decay chain** using **Cython** for performance optimization. It benchmarks the Cython implementation against various other languages and libraries such as Python, Julia, C++, MATLAB, Mathematica, Numba, and R.

## Table of Contents

- [Introduction](#introduction)
- [Project Aims](#project-aims)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Radioactive decay is a random process that can be modeled using probabilistic techniques. In this project, we use Monte Carlo methods to simulate the decay of radioactive elements over time. To improve performance, we have implemented the simulation in **Cython** and benchmarked it against other programming languages and libraries.

## Project Aims

The main goals of this project are:
1. To model radioactive decay chains using Monte Carlo simulations.
2. To optimize the performance using Cython.
3. To benchmark the performance against various programming languages and libraries.

## Features

- Monte Carlo simulation of radioactive decay chains.
- Optimized simulation using Cython.
- Performance benchmarking across multiple languages and libraries.
- Detailed documentation and code explanations.
- Examples of how to use the simulation and benchmarking.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/radioactive-decay-cython.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. To build the Cython code, run:
    ```bash
    python setup.py build_ext --inplace
    ```

## Usage

### Running the Simulation

To run the Monte Carlo simulation of radioactive decay, execute the following command:

```bash
python simulate_decay.py --elements Uranium Thorium --steps 1000

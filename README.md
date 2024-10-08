<h1>Radioactive Decay Chain Probability Estimation using Cython and Monte Carlo Simulations</h1>

<p>This repository contains the implementation of a Monte Carlo simulation to model radioactive decay chains. The project is optimized using <strong>Cython</strong> for performance and benchmarked against implementations in other programming languages such as Python (Numba), Julia, C++, MATLAB, Mathematica, and R.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#requirements">Requirements</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a>
    <ul>
      <li><a href="#running-the-cython-simulation">Running the Cython Simulation</a></li>
      <li><a href="#running-other-implementations">Running Other Implementations</a></li>
    </ul>
  </li>
  <li><a href="#benchmarking">Benchmarking</a></li>
  <li><a href="#results-and-analysis">Results and Analysis</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
  <li><a href="#references">References</a></li>
</ul>

<hr>

<h2 id="introduction">Introduction</h2>
<p>Radioactive decay is a random process by which unstable atomic nuclei lose energy by emitting radiation. A decay chain is a series of successive radioactive decays that certain isotopes undergo until they reach a stable isotope.</p>
<p>This project models a simple radioactive decay chain using Monte Carlo simulations. The simulation is implemented in <strong>Cython</strong> to leverage its performance advantages over pure Python. Additionally, we compare the performance of the Cython implementation with versions written in other languages and libraries, including Julia, C++, MATLAB, Mathematica, Numba, and R.</p>

<hr>

<h2 id="features">Features</h2>
<ul>
  <li><strong>Monte Carlo Simulation</strong> of a radioactive decay chain (A → B → C).</li>
  <li><strong>Cython Implementation</strong> for optimized performance.</li>
  <li>Implementations in <strong>multiple languages</strong> for benchmarking:
    <ul>
      <li>Pure Python</li>
      <li>Python with Numba</li>
      <li>Julia</li>
      <li>C++</li>
      <li>MATLAB</li>
      <li>Mathematica</li>
      <li>R</li>
    </ul>
  </li>
  <li><strong>Benchmarking Scripts</strong> to compare execution time and memory usage.</li>
  <li><strong>Visualization</strong> of simulation results using Matplotlib.</li>
</ul>

<hr>

<h2 id="requirements">Requirements</h2>
<ul>
  <li><strong>Python 3.6+</strong></li>
  <li><strong>Cython</strong></li>
  <li><strong>NumPy</strong></li>
  <li><strong>Matplotlib</strong></li>
</ul>
<p>Additional requirements for other implementations:</p>
<ul>
  <li><strong>Julia</strong> (for the Julia version)</li>
  <li><strong>C++ compiler</strong> (e.g., g++)</li>
  <li><strong>MATLAB</strong></li>
  <li><strong>Mathematica</strong></li>
  <li><strong>R</strong></li>
</ul>

<hr>

<h2 id="installation">Installation</h2>

<h3>Clone the Repository</h3>
<pre><code>git clone https://github.com/yourusername/decay-chain-simulation.git
cd decay-chain-simulation
</code></pre>

<h3>Set Up Python Environment</h3>
<p>It's recommended to use a virtual environment.</p>
<pre><code>python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
</code></pre>

<h3>Install Python Dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>
<p>This will install <code>Cython</code>, <code>NumPy</code>, and <code>Matplotlib</code>.</p>

<hr>

<h2 id="usage">Usage</h2>

<h3 id="running-the-cython-simulation">Running the Cython Simulation</h3>

<h4>1. Compile the Cython Code</h4>
<p>Navigate to the <code>cython</code> directory:</p>
<pre><code>cd cython
</code></pre>
<p>Run the setup script:</p>
<pre><code>python setup.py build_ext --inplace
</code></pre>

<h4>2. Run the Simulation</h4>
<p>Execute the <code>main.py</code> script:</p>
<pre><code>python main.py
</code></pre>

<p>This script will:</p>
<ul>
  <li>Run the simulation using the compiled Cython module.</li>
  <li>Plot the results showing the number of nuclei of each isotope over time.</li>
</ul>

<h4>3. Output</h4>
<p>A plot will be displayed showing the decay curves for Isotopes A, B, and C.</p>

<h3 id="running-other-implementations">Running Other Implementations</h3>
<p>Instructions for running other implementations are provided in their respective directories:</p>
<ul>
  <li><strong>Python (Numba):</strong> Located in the <code>python_numba</code> directory.</li>
  <li><strong>Julia:</strong> Located in the <code>julia</code> directory.</li>
  <li><strong>C++:</strong> Located in the <code>cpp</code> directory.</li>
  <li><strong>MATLAB:</strong> Located in the <code>matlab</code> directory.</li>
  <li><strong>Mathematica:</strong> Located in the <code>mathematica</code> directory.</li>
  <li><strong>R:</strong> Located in the <code>r</code> directory.</li>
</ul>
<p>Please refer to the <code>README.md</code> files in each directory for detailed instructions.</p>

<hr>

<h2 id="benchmarking">Benchmarking</h2>

<h3>Benchmarking Parameters</h3>
<ul>
  <li><strong>Simulation Parameters:</strong></li>
  <ul>
    <li>λ₁ = 0.01 (Decay constant for A → B)</li>
    <li>λ₂ = 0.005 (Decay constant for B → C)</li>
    <li>N₀ = 10,000 (Initial number of Isotope A nuclei)</li>
    <li>T = 500 (Total time)</li>
    <li>Δt = 1 (Time step)</li>
  </ul>
  <li><strong>System Specs:</strong></li>
  <ul>
    <li>CPU: Quad-core processor</li>
    <li>RAM: 16 GB</li>
  </ul>
</ul>

<h3>Running the Benchmark</h3>
<p>Use the <code>benchmark.py</code> script to run benchmarks on all implementations.</p>
<pre><code>python benchmark.py
</code></pre>
<p>This script will:</p>
<ul>
  <li>Run each implementation.</li>
  <li>Measure execution time.</li>
  <li>Collect results for comparison.</li>
</ul>

<h3>Results</h3>
<p>Benchmarking results will be displayed in the console and saved to <code>benchmark_results.txt</code>.</p>

<hr>

<h2 id="results-and-analysis">Results and Analysis</h2>

<h3>Execution Time Comparison</h3>

<table>
  <thead>
    <tr>
      <th>Language</th>
      <th>Execution Time (seconds)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>C++</td><td>1.2</td></tr>
    <tr><td>Cython</td><td>1.3</td></tr>
    <tr><td>Julia</td><td>1.5</td></tr>
    <tr><td>Python + Numba</td><td>1.7</td></tr>
    <tr><td>MATLAB</td><td>4.5</td></tr>
    <tr><td>R</td><td>6.0</td></tr>
    <tr><td>Pure Python</td><td>25.0</td></tr>
  </tbody>
</table>

<h3>Analysis</h3>
<ul>
  <li><strong>C++</strong> is the fastest due to its compiled nature and low-level optimizations.</li>
  <li><strong>Cython</strong> significantly improves performance over pure Python, closely matching C++.</li>
  <li><strong>Julia</strong> offers competitive performance with simpler syntax for mathematical computations.</li>
  <li><strong>Numba</strong> provides JIT compilation in Python, enhancing performance.</li>
  <li><strong>MATLAB</strong> and <strong>R</strong> are slower but acceptable for smaller simulations.</li>
  <li><strong>Pure Python</strong> is the slowest due to interpreter overhead and dynamic typing.</li>
</ul>

<h3>Memory Usage</h3>
<ul>
  <li>All implementations have similar memory footprints due to the storage of arrays.</li>
  <li>Memory optimizations include preallocating arrays and using appropriate data types.</li>
</ul>

<hr>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please read <code>CONTRIBUTING.md</code> for guidelines.</p>

<hr>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License - see the <code>LICENSE</code> file for details.</p>

<hr>

<h2 id="acknowledgments">Acknowledgments</h2>
<ul>
  <li><strong>Cython Documentation:</strong> <a href="https://cython.readthedocs.io">https://cython.readthedocs.io</a></li>
  <li><strong>NumPy Documentation:</strong> <a href="https://numpy.org/doc/">https://numpy.org/doc/</a></li>
  <li><strong>Monte Carlo Methods:</strong> <a href="https://en.wikipedia.org/wiki/Monte_Carlo_method">Wikipedia - Monte Carlo Method</a></li>
  <li><strong>Radioactive Decay Chains:</strong> <a href="https://en.wikipedia.org/wiki/Decay_chain">Wikipedia - Decay Chain</a></li>
  <li><strong>Julia Language:</strong> <a href="https://julialang.org">https://julialang.org</a></li>
</ul>

<hr>

<h2 id="references">References</h2>
<ul>
  <li>Bateman, H. (1910). <em>The solution of a system of differential equations occurring in the theory of radioactive transformations</em>.</li>
  <li><em>Monte Carlo Methods in Statistical Physics</em>.</li>
  <li><em>Benchmarking Python Scientific Computing</em>.</li>
</ul>

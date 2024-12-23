# Differential Evolution Algorithm: A Practical Implementation

## Overview

Differential Evolution (DE) is a robust and efficient optimization algorithm widely used for solving complex optimization problems. DE belongs to the family of evolutionary algorithms and operates on a population of candidate solutions. Through iterative processes of mutation, crossover, and selection, it evolves the population towards an optimal solution.

Key advantages of DE include:
- **Versatility:** Applicable to a wide range of optimization problems, including non-linear, non-differentiable, and multi-modal problems.
- **Simplicity:** Relies on a few control parameters and straightforward operations.
- **Efficiency:** Achieves high-quality solutions with relatively low computational costs.

This repository implements the DE algorithm for two scenarios:
1. Optimization of the Ackley function (a well-known benchmark for optimization algorithms).
2. Optimization of a wind turbine design for cost and performance.

---

## Algorithm Description

### Main Steps
1. **Initialization:**
   - Generate a random population of candidate solutions within specified bounds.
2. **Mutation:**
   - Create a mutant vector by combining three random individuals in the population with a scaling factor (F).
3. **Crossover:**
   - Combine the mutant vector with the target vector to form a trial vector.
4. **Selection:**
   - Compare the fitness of the trial vector with the target vector and retain the better one.
5. **Termination:**
   - Repeat steps 2-4 for a predefined number of iterations or until convergence.

### Control Parameters
- **Population Size (NP):** Number of candidate solutions.
- **Scaling Factor (F):** Controls the differential variation during mutation.
- **Crossover Rate (CR):** Probability of inheriting genes from the mutant vector.
- **Bounds:** Defines the solution space for each dimension.

---

## Files and Functions

### `ackley.py`
This script focuses on optimizing the 2D Ackley function, a non-linear and multi-modal test function commonly used to evaluate optimization algorithms.

#### Key Features:
- **Ackley Function Implementation:** A mathematical representation of the optimization problem.
- **Population Initialization:** Random generation of candidate solutions within defined bounds.
- **Mutation and Crossover Operators:** Implementation of DE operations.
- **Visualization:**
  - Contour plots for population distribution.
  - 3D surface plots of the Ackley function.
  - Iteration-by-iteration visualizations of population evolution.

#### Usage:
Run the script to observe how DE finds the global minimum of the Ackley function:
```bash
python ackley.py
```

---

### `ackley2.py`
This script is a variation of `ackley.py` with additional features and streamlined visualizations. It includes:
- Simplified DE operations for testing and benchmarking.
- Visualization of convergence and population dynamics over iterations.

#### Usage:
Execute the script to see the optimization process in action:
```bash
python ackley2.py
```

---

### `main2.py`
This script demonstrates the application of DE to optimize the design of a wind turbine for minimal cost while achieving a target power output.

#### Key Features:
- **Problem Definition:**
  - Parameters: Blade length, axial speed coefficient, blade thickness, blade angle, and blade count.
  - Objective: Minimize total cost (production, maintenance, and penalty for deviation from target power).
- **Fitness Function:** Combines cost components into a single optimization objective.
- **Visualization:**
  - Convergence of costs.
  - Evolution of parameter values.
  - Breakdown of cost components.
  - Simplified turbine visualization.

#### Usage:
Run the script to optimize turbine parameters:
```bash
python main2.py
```

---

## Installation and Dependencies
This project requires Python 3.7+ and the following libraries:
- `numpy`
- `matplotlib`
- `seaborn`

Install dependencies using:
```bash
pip install numpy matplotlib seaborn
```

---

## Examples
### Ackley Function Optimization
- Visualizes the DE process in solving a 2D benchmark optimization problem.

### Wind Turbine Design
- Demonstrates real-world application with multi-criteria optimization.

---

## Results
- **Ackley Function:** DE effectively locates the global minimum with a clear visual representation of the optimization process.
- **Wind Turbine:** Achieves an optimal design balancing cost and power output, visualized through convergence and parameter graphs.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contribution
Feel free to fork this repository, open issues, or submit pull requests to improve the implementation or add new features.


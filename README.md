# Genetic-Algorithms

**Solving real-world optimization problems by adapting a Genetic Algorithm (GA) framework in Python.**

This repository demonstrates how a **pre-existing GA skeleton** can be extended to tackle combinatorial optimization problems, including:

- **Machine Servicing Problem** – Maximize total machine output while satisfying periodic service requirements.  
- **Traveling Salesman Problem (TSP)** – Find the shortest route visiting all cities exactly once.  
- **Knapsack Problem** – Maximize total value of items without exceeding weight constraints.  

## Approach

Instead of implementing a GA from scratch, this project **leverages a GA skeleton** that provides:

- **Population management**  
- **Selection mechanisms** (Roulette Wheel, Tournament, Rank-Based)  
- **Crossover and mutation operations**  

The framework was **extended** to each problem by:

- Designing **problem-specific chromosomes**  
- Implementing **fitness functions** with constraints and penalties  
- Adjusting **crossover and mutation strategies** to fit the problem  

## Key Concepts Applied

- **GA Framework:** Initialization, Selection, Crossover, Mutation, Iteration  
- **Chromosome Design:** Encodes problem-specific decisions  
- **Fitness Evaluation:** Computes solution quality while handling constraints  
- **Python OOP:** Extending base classes for problem-specific implementations  

## Tech Stack

- **Language:** Python 3  
- **Libraries:** `random`, `itertools`, `numpy` (optional for some problems)  

## Learning Outcomes

- How to **adapt a generic GA skeleton** to multiple optimization problems  
- Designing **fitness functions** that balance objectives and constraints  
- Applying GA to **real-world scheduling**

## Problems:
[Machine Service Problem](MachineServiceProblem.py)

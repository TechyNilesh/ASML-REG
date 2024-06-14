# ASML-REG
Automated Machine Learning for Data Stream Regression

# ASML-REG

Automated Machine Learning for Data Stream Regression

### Dataset Overview

[Table 1](#reg_bench_dataset) provides an overview of the datasets used for evaluating the performance of the ASML_REG against the baseline algorithm on data stream regression tasks. These datasets include both real-world and synthetic datasets, covering a wide range of features and instances. 

#### Table 1: Dataset Overview {#reg_bench_dataset}

| **Datasets**     | **N_Features** | **N_Instances** | **Source**  |
|------------------|----------------|-----------------|-------------|
| Ailerons         | 40             | 13750           | Synthetic   |
| Elevators        | 18             | 16599           | Synthetic   |
| Fried            | 10             | 40768           | Synthetic   |
| Friedman GRA     | 10             | 100000          | Synthetic   |
| Friedman GSG     | 10             | 100000          | Synthetic   |
| Friedman LEA     | 10             | 100000          | Synthetic   |
| Hyper (A)        | 10             | 500000          | Synthetic   |
| Abalone          | 8              | 4977            | Real        |
| Bike             | 12             | 17379           | Real        |
| Cpu Activity     | 22             | 8192            | Real        |
| House8L          | 8              | 22784           | Real        |
| Kin8nm           | 9              | 8192            | Real        |
| Metro Traffic    | 7              | 48204           | Real        |
| White Wine       | 11             | 4898            | Real        |

Most of these datasets are standard benchmarks. For instance, the Abalone dataset, sourced from a non-machine-learning research paper [Nash, 1994](#nash1994population), aims to predict the age of abalones based on physical measurements. The Fried dataset [Friedman, 1991](#friedman1991multivariate) is synthetic and uses a highly non-linear formula, incorporating irrelevant features to test the robustness of regressors:

\[ y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \sigma(0, 1) \]

The HyperA dataset [Bifet, 2011](#bifet2011moa) generates a hyperplane in a \(d\)-dimensional space, where the goal is to predict the distance from randomly generated data points to the hyperplane. This dataset is particularly useful for assessing drift detection ability, as it simulates concept drift at specific intervals (125K, 250K, and 375K instances).

The Friedman dataset [Ikonomovska, 2011](#ikonomovska2011learning), another synthetic dataset, is designed to simulate different types of concept drift. Each observation consists of 10 features, with the first 5 being relevant. The target is defined by different functions depending on the type of drift:

- **LEA**: Local Expanding Abrupt drift with three points of abrupt change.
- **GRA**: Global Recurring Abrupt drift with two points of concept drift.
- **GSG**: Global and Slow Gradual drift with gradual change during transition windows.

These synthetic datasets, along with real-world datasets, ensure a comprehensive evaluation of the algorithm's performance in diverse scenarios.


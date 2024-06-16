# ASML-REG

Automated Machine Learning for Data Stream Regression

### Data Streams Regression Benchmarking Dataset 

The dataset files can be found in the GitHub repository folder `RDatasets`.

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

Most of these datasets are standard benchmarks. For instance, the Abalone dataset, sourced from a non-machine-learning research paper [Nash, 1994](https://www.researchgate.net/profile/Warwick-Nash/publication/287546509_7he_Population_Biology_of_Abalone_Haliotis_species_in_Tasmania_I_Blacklip_Abalone_H_rubra_from_the_North_Coast_and_Islands_of_Bass_Strait/links/5d949460458515202b7bf592/7he-Population-Biology-of-Abalone-Haliotis-species-in-Tasmania-I-Blacklip-Abalone-H-rubra-from-the-North-Coast-and-Islands-of-Bass-Strait.pdf), aims to predict the age of abalones based on physical measurements. The Fried dataset [Friedman, 1991](https://doi.org/10.1214/aos/1176347963) is synthetic and uses a highly non-linear formula, incorporating irrelevant features to test the robustness of regressors:

\[ y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \sigma(0, 1) \]

The HyperA dataset [Bifet, 2011](https://proceedings.mlr.press/v11/bifet10a.html) generates a hyperplane in a \(d\)-dimensional space, where the goal is to predict the distance from randomly generated data points to the hyperplane. This dataset is particularly useful for assessing drift detection ability, as it simulates concept drift at specific intervals (125K, 250K, and 375K instances).

The Friedman dataset [Ikonomovska, 2011](https://doi.org/10.1007/s10618-010-0201-y), another synthetic dataset, is designed to simulate different types of concept drift. Each observation consists of 10 features, with the first 5 being relevant. The target is defined by different functions depending on the type of drift:

- **LEA**: Local Expanding Abrupt drift with three points of abrupt change.
- **GRA**: Global Recurring Abrupt drift with two points of concept drift.
- **GSG**: Global and Slow Gradual drift with gradual change during transition windows.

These synthetic datasets, along with real-world datasets, ensure a comprehensive evaluation of the algorithm's performance in diverse scenarios.

# Requirements

# Run Code
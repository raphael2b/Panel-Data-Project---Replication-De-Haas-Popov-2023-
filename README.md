# Panel Data Econometrics: Replication Project
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white) ![Stata](https://img.shields.io/badge/Stata-0A4B78?style=flat&logo=stata&logoColor=white)

## 📋 Project Overview
This project, developed for the **M2 Panel Data Course**, focuses on the replication of peer-reviewed research. The analysis explores variables varying across both time and individuals, utilizing transformations to isolate within-group and between-group variations.

## 👨‍💻 Replication Assignment: De Haas & Popov (2023)
**Paper**: *De Haas, Popov (2023), Finance and Green Growth*, published in the **Economic Journal**.
**Dataset**: Sector-level panel data.
**Core Task**: Investigate the "finance-growth nexus" by replicating and extending the original authors' methodology using diverse software environments.

Part of this project involves using an LLM to replicate the original code I wrote in other languages and to correct it. You will find an R, STATA files that does the same thing than the code I wrote in Python. To replicate the original paper from De Haas & Popov, I have to answer 28 questions about the transformation and analysis of data.

## 💾 Python Ressources
Pandas: Essential for managing the multi-indexed structure of panel data (Country-Year) and calculating the "Within" and "Between" components for each variable.

NumPy: Used for performing first-difference transformations of x and ensuring proper handling of missing values (dots/NaN) during individual transitions in the stacked dataset.

Statsmodels: Utilized for estimating Two-Way Fixed Effects (TWFE) by regressing within-transformed variables on time dummies. It also provides tools for diagnostic unit root tests on panel series.

Linearmodels: The specialized library used to implement standard panel benchmarks, including Fixed Effects (FE), Random Effects (RE) with Mundlak terms, and First Differences (FD) estimators.

SciPy (stats): Employed for analyzing the distributions of transformed variables, specifically comparing Kurtosis, Skewness, and Kernel Continuous Approximations against the Normal Law.

Matplotlib.pyplot: The core tool for generating comparative plots of Between, Within, and Two-Way Fixed Effects distributions, as well as plotting Impulse Response Functions (IRF) for dynamic models.

Seaborn: Integrated for high-level visualization, such as bivariate clouds of points with regression lines and marginal distributions to check for high-leverage observations
##
As a student if you have any remarks or comments about this work contact me ! 😁

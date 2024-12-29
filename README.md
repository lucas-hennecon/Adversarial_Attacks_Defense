# Adversarial Attacks and Defense Mechanisms

This repository contains implementations of **adversarial attacks** and **defense mechanisms**, developed for **Data Science Lab Project 3 (IASD)** in collaboration with **Nan AN** and **Hangyue Zhao**. The project explores model robustness against adversarial examples and strategies to mitigate their impact. For a detailed presentation of the project and results, refer to the `Report.pdf` file.

## 📜 Overview
### Attacks:
- **FGSM**: Fast Gradient Sign Method.
- **PGD (L∞ & L2)**: Projected Gradient Descent.
- **MI-FGSM**: Momentum Iterative FGSM.
- **Diverse Inputs Attack**: M-DI2-FGSM.

### Defenses:
- **Adversarial Training**: Training with adversarial examples.
- **Mixed Adversarial Training (MAT)**: Combines L∞ and L2 examples.

## 📂 Project Structure

- **`models/`**: Directory containing saved models
  - **`mymodel.pth`**: model trained with mixed adversarial training.
- **`adversarial_training.py`**: Implements adversarial training as a defense mechanism.
- **`dim_attack.py`**: Diverse Inputs Iterative FGSM attack implementation.
- **`fgsm_attack.py`**: Fast Gradient Sign Method (FGSM) attack implementation.
- **`mat_rand_training.py`**: Implements Mixed Adversarial Training (MAT) defense.
- **`mim_attack.py`**: Momentum Iterative FGSM attack implementation.
- **`model.py`**: Utilities for loading and managing models.
- **`pgd_attack.py`**: PGD attack implementation for L∞ and L2 norms.
- **`test_attack_defense.py`**: Script for testing attacks and defenses.
- **`utils.py`**: Helper functions for evaluation metrics and visualizations.
- **`README.md`**: Project documentation.
- **`Report.pdf`**: Detailed project report with methodologies and results.

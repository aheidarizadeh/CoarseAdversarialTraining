# CoarseAdversarialTraining
Implementation of Coarse Adversarial Training (CAT) for improving robustness to coarse-level adversarial attacks. Contains training pipelines, CASR evaluation, and visualization scripts.

## Overview
This repository provides code to train and evaluate three regimes on CIFAR-10 and FMNIST:  

- **NT (Natural Training):** standard training on clean data.  
- **SAT (Standard Adversarial Training):** PGD-based adversarial training.  
- **CAT (Coarse Adversarial Training):** adversarial training on coarse labels.  

We study robustness against coarse adversarial perturbations, where adversarial examples change the predicted **coarse category** (e.g., vehicles vs. mammals) rather than only fine labels.

## Features
- CIFAR-10 training pipelines for NT, SAT (PGD-10), and CAT (PGD-10).
%- FMNIST training with NT, SAT (FGSM_r), and CAT (FGSM_r).
- Coarse mapping utilities for semantic or random groupings.
- Evaluation code for CASR-vs-ε curves.
- Scripts to generate adversarial examples and visualize minimal perturbations.

# 🧬 CCIR Program: Advanced Biomedical Imaging and Deep Learning 🧬

Welcome to the **CCIR Program** repository, a collection of scripts and codes for the Computational Cancer Imaging Research (CCIR) program. This repository includes various homework assignments, foundation models, and deep learning models. Here you'll find detailed instructions on setting up the environment, running the scripts, and utilizing the models effectively.

## Table of Contents
- [About the Repository](#about-the-repository)
- [Scripts Overview](#scripts-overview)
  - [Script Code 1: Tissue Classification with Deep Learning](#script-code-1-tissue-classification-with-deep-learning)
  - [Script Code 2: Tissue Classification with HOG Features and SVM](#script-code-2-tissue-classification-with-hog-features-and-svm)
  - [Script Code 3: Segment Anything Model (SAM v1)](#script-code-3-segment-anything-model-sam-v1)
  - [Script Code 4: Segment Anything Model (SAM v2)](#script-code-4-segment-anything-model-sam-v2)
  - [Script Code 5: Foundation Model for Cancer Imaging Biomarker (CT Images)](#script-code-5-foundation-model-for-cancer-imaging-biomarker-ct-images)
  - [Script Code 6: Homework on Deep Learning (Coming Soon)](#script-code-6-homework-on-deep-learning-coming-soon)
  - [Script Code 7: Another Homework on Deep Learning (Coming Soon)](#script-code-7-another-homework-on-deep-learning-coming-soon)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Scripts](#how-to-run-the-scripts)
- [Acknowledgements](#acknowledgements)

## About the Repository

This repository is dedicated to the **CCIR Program**, encompassing a series of scripts designed for advanced biomedical imaging and deep learning tasks. The scripts include homework assignments, foundation models, and various deep learning applications in medical imaging.

## Scripts Overview

### Script Code 1: Tissue Classification with Deep Learning

**Description**: Train a model for classifying tissue samples into benign vs. malign using a deep learning architecture.

**Code**: [script_dem_1.ipynb](script_code_1/script_dem_1.ipynb)

**Environment Setup**:
```bash
conda create -n image_processing_env python=3.8
conda activate image_processing_env
conda install -c conda-forge pandas scikit-learn matplotlib numpy pillow tensorflow
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=image_processing_env --display-name "Python (image_processing_env)"

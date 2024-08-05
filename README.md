
# 🧬 CCIR Program: Advanced Biomedical Imaging and Deep Learning Demos 🧬

[![Python](https://img.shields.io/badge/Python-3.8%2C%203.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Conda](https://img.shields.io/badge/Conda-environment-blue)](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

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
```

### Script Code 2: Tissue Classification with HOG Features and SVM

**Description**: Train a model for classifying tissue samples into benign vs. malign using HOG features and an SVM classifier.

**Code**: [script_dem_2.ipynb](script_code_2/script_dem_2.ipynb)

**Environment Setup**:
```bash
conda create -n svm_image_env python=3.8
conda activate svm_image_env
conda install -c conda-forge numpy matplotlib opencv jupyter
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=svm_image_env --display-name "Python (svm_image_env)"
jupyter notebook
```

### Script Code 3: Segment Anything Model (SAM v1)

**Description**: Implement the Segment Anything Model (SAM v1) for image segmentation tasks.

**Code**: [gui.py](script_code_3/gui.py)

**Environment Setup**:
```bash
conda create -n medsam python=3.10 -y
conda activate medsam
pip3 install torch torchvision torchaudio
cd script_code_3
pip install -e .
```

### Script Code 4: Segment Anything Model (SAM v2)

**Description**: An updated version of the Segment Anything Model (SAM v2) for enhanced image segmentation capabilities.

**Code**: [predictor_example.ipynb](script_code_4/predictor_example.ipynb)

**Environment Setup**:
```bash
conda create -n medsam2d python=3.10 -y
conda activate medsam2d
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install numpy
pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=medsam2d --display-name "Python (medsam2d)"
pip install pydrive
pip install albumentations
pip install -U numpy
conda update albumentations scipy scikit-image
conda install -c conda-forge cupy
# download model from here: https://zenodo.org/records/10528450/files/model_weights.torch?download=1
```

### Script Code 5: Foundation Model for Cancer Imaging Biomarker (CT Images)

**Description**: Utilize the Foundation Model for Cancer Imaging Biomarker (CT Images) developed by AIM-Harvard.

**Code**: Coming Soon

**Environment Setup**:
```bash
conda create -n foundation_model_ct python=3.8
conda activate foundation_model_ct
pip install foundation-cancer-image-biomarker
cd /mnt/Data1/GitHub/Code_Demo_CCIR_2024/script_code_5/
pip install -r additional_requirements.txt
pip install -U "huggingface_hub[cli]"
```

**Model Download**:
```bash
huggingface-cli download surajpaib/fmcib --local-dir '/mnt/movs/Downloads/pretrain_model/'
# Or from here
https://zenodo.org/records/10528450/files/model_weights.torch?download=1
```

**Inference**:
```bash
lighter predict --config_file ./experiments/inference/get_predictions.yaml
```

### Script Code 6: Homework on Deep Learning (Coming Soon)

### Script Code 7: Another Homework on Deep Learning (Coming Soon)

## Installation and Setup

Follow the environment setup instructions provided for each script to install the necessary dependencies and create the required Conda environments.

## How to Run the Scripts

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CCIR-Program.git
   cd CCIR-Program
   ```

2. Navigate to the respective script directory and follow the setup instructions provided.

3. Launch Jupyter Notebook or run the Python scripts as indicated.

## Acknowledgements

Special thanks to the AIM-Harvard team and the contributors of the various libraries and models used in this repository.

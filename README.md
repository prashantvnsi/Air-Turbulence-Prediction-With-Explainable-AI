# Air-Turbulence-Prediction-With-Explainable-AI
Transformer models to predict air turbulence using physical data and using explainable AI to explain the result.

This repository contains the complete source code for the master's project titled "Explainable AI for Turbulence Prediction: Leveraging Transformer Architectures to Enhance Trust and Usability in Aviation Safety."

The implementation compares two Transformer-based models—Vision Transformer (ViT) and Perceiver IO—on turbulence prediction using high-resolution atmospheric simulation data.

🔍 Repository Contents
File	Description
DataGeneration.ipynb	Preprocessing pipeline to extract 32×32 patches from the SABL2048HIGH dataset, compute physical variables (velocity, temperature, pressure), and apply the composite labelling strategy. The output is a .pt file used for model training.
VisionTransformerImplementation.ipynb	Implements the ViT model using the HuggingFace vit-base-patch16-224-in21k checkpoint. Includes fine-tuning, evaluation, and SHAP-based explanation generation.
PerceiverIOImplementation.ipynb	Implements the Perceiver IO model using the pretrained deepmind/vision-perceiver-learned checkpoint. Also includes Optuna-based hyperparameter tuning, classification evaluation, and attribution mapping.
README.md	This file. Contains instructions, explanations, and additional notes on environment setup and dataset usage.

🧠 Dataset
The full training dataset used for this project is stored as a PyTorch tensor in the file:

Copy
Edit
turbulence_dataset_sabl2048high_composite_label.pt
However, this file is approximately 432 MB, which exceeds GitHub's file size limit of 100 MB. As a result, it is not included in the repository.

If you would like to use or inspect the dataset, you can:

Regenerate it using DataGeneration.ipynb, which connects to the Johns Hopkins Turbulence Database (JHTDB).

Or, if you already have access to the .pt file, manually move it into your working directory before training.

⚠️ For users unfamiliar with Git LFS (Large File Storage), see Git LFS setup if you still want to track large binary files in the future.

# 102-Category Flower Classification Using ResNet18 and MLOps

## Project Description  
This project implements a complete MLOps pipeline for classifying 102 categories of flowers using deep learning techniques. The model is built using PyTorch Lightning and is trained on the **Oxford 102 Flower Dataset**, which contains a diverse set of flower species with variations in lighting, pose, and background. The pipeline integrates key MLOps tools for experiment tracking, data versioning, model deployment, and monitoring.

## Objective  
Our primary objective is to develop a robust and reproducible deep learning pipeline that can accurately classify flower species from raw images. The project covers the full ML lifecycle, including data preprocessing, model training, evaluation, deployment, and continuous integration.

## Framework and Integration  
We leverage the following tools and frameworks in our pipeline:
- **PyTorch Lightning** for scalable and modular training  
- **DVC** (Data Version Control) for dataset and model versioning  
- **Weights & Biases (W&B)** for experiment tracking  
- **Docker** for environment consistency and containerized deployment  
- **FastAPI** to serve the model through a REST API  
- **Streamlit** for building an interactive frontend to visualize predictions  

## Data Collection and Initial Dataset  
The dataset used is the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). It includes **8,189 images** across **102 flower categories**, captured under various real-world conditions.

- Raw images were used directly as input.  
- Preprocessing was applied before training.

### Dataset Details:
- 102 flower classes  
- 8,189 total images  
 

## Model Architecture  
We employ the **ResNet18** convolutional neural network, a proven architecture for image classification tasks. It has been fine-tuned to adapt to the flower dataset using transfer learning techniques and trained using PyTorch Lightning for clarity and scalability.

## Metrics Tracked  
We tracked the following key metrics using W&B:
- **Training Loss**  
- **Validation Loss**  
- **Accuracy**  
- **Steps/Epochs**  

These metrics are essential for monitoring training progress, detecting overfitting, and optimizing model performance.

## Deployment and Automation  
- **FastAPI** is used for building a RESTful inference API  
- **Streamlit** provides an interactive user interface to visualize and test predictions  
- **Docker** ensures reproducible builds and consistent environments  

## Team Members  
- Praveen Titus Francis  
- Dileep Vemuri  
- Ali Najibpour Nashi 

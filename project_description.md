# 102-Category Flower Classification Using ResNet18 and MLOps

## Project Description  
This project implements a complete MLOps pipeline for classifying 102 categories of flowers using deep learning techniques. The model is built using PyTorch Lightning and is trained on the **Oxford 102 Flower Dataset**, which contains a diverse set of flower species with variations in lighting, pose, and background. The pipeline integrates key MLOps tools for experiment tracking, data versioning, model deployment, and monitoring.

### Link to the deployed app: https://frontend-660622539098.europe-west1.run.app/

## Objective  
Our primary objective is to develop a robust and reproducible deep learning pipeline that can accurately classify flower species from raw images. The project covers the full ML lifecycle, including data preprocessing, model training, evaluation and deployment.

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

## Deployment 
- **FastAPI** is used for building a RESTful inference API  
- **Streamlit** provides an interactive user interface to visualize and test predictions  
- **Docker** ensures reproducible builds and consistent environments  

## Team Members  
- Praveen Titus Francis  
- Dileep Vemuri  
- Ali Najibpour Nashi

```
mlopspj/
    ├── .dvc/                        <- DVC cache and metadata
    ├── .dvcignore                   <- Ignore patterns for DVC tracking
    ├── .gitignore                   <- Ignore rules for Git
    ├── .python-version             <- Python version pin (3.12)
    ├── backend_requirements.txt    <- Backend dependencies
    ├── frontend_requirements.txt   <- Frontend dependencies
    ├── preprocess_requirements.txt <- Preprocessing dependencies
    ├── train_requirements.txt      <- Training dependencies
    ├── pyproject.toml              <- Project configuration (likely Poetry)
    ├── LICENSE                     <- Open-source license
    ├── README.md                   <- Main project README (with Exam questions)
    ├── project_description.md      <- Detailed description about the project

    ├── configs/                    <- Configuration files for modules or pipelines
    
    ├── data/                       <- Versioned data folder (tracked with DVC)
    │   ├── .gitignore              <- Ignore file inside data/
    │   ├── flower_labels.csv.dvc   <- DVC-tracked label data
    │   ├── labels.csv.dvc          <- DVC-tracked label CSV
    │   └── raw_images.dvc          <- DVC-tracked raw images
    
    ├── dockerfiles/                <- Dockerfiles for each module
    │   ├── backend.dockerfile
    │   ├── frontend.dockerfile
    │   ├── preprocess.dockerfile
    │   └── train.dockerfile
    
    ├── src/
    │   └── flowerclassif/          <- Core source code module
    │       ├── __init__.py         <- Makes it a Python package
    │       ├── __pycache__/        <- Compiled Python files
    │       ├── backend.py          <- Backend application logic
    │       ├── frontend.py         <- Frontend logic and interface
    │       ├── preprocess.py       <- Data preprocessing script
    │       └── train.py            <- Training script
```

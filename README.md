# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `train.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the  `preprocess_requirements.txt`, `train_requirements.txt` `backend_requirements.txt`,  and `frontend_requirements.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [x] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

### Question 2
> **Enter the study number for each member in the group**
>
> Answer:
> Praveen Titus Francis : 12837557
> 
> Dileep Vemuri : 12818965
> 
> Ali Najibpour Nashi : 12644070

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Answer: For this project, we used ResNet-18, a pre-trained deep convolutional neural network from the torchvision.models library. Although PyTorch was introduced in the course, using pre-trained architectures like ResNet-18 was not covered, making it a suitable third-party addition. ResNet-18’s skip connections enable efficient training and strong performance on image classification tasks. Leveraging transfer learning with this model allowed us to achieve high accuracy while reducing training time and computational cost. It also helped streamline the development process, allowing us to focus more on the MLOps pipeline, including data handling, experiment tracking, and deployment. Overall, ResNet-18 added great value.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer:We used a virtual environment to isolate the project's dependencies from the system-wide Python packages, ensuring a clean and conflict-free setup. To manage dependencies across different stages of the project, we maintained separate requirement files: preprocess_requirements.txt, train_requirements.txt, backend_requirements.txt, and frontend_requirements.txt. These files allow new team members to easily install the necessary packages for each component.

>Additionally, we created Dockerfiles to containerize the application, ensuring consistent environments across development, testing, and deployment. Instead of manually installing dependencies, users can simply build and run the Docker containers, which encapsulate all setup steps for each module.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Answer: Project structure: used Cookie-cutter template, and kept only the relevant folder.
```
mlopspj/
    ├── .dvc/                        <- DVC cache and metadata
    ├── .dvcignore                   <- Ignore patterns for DVC tracking
    ├── .gitignore                   <- Ignore rules for Git
    ├── .python-version             <- Python version pin (3.12)
    ├── .ruff_cache/                <- Ruff linter/cache directory
    ├── backend_requirements.txt    <- Backend dependencies
    ├── frontend_requirements.txt   <- Frontend dependencies
    ├── preprocess_requirements.txt <- Preprocessing dependencies
    ├── train_requirements.txt      <- Training dependencies
    ├── pyproject.toml              <- Project configuration (likely Poetry)
    ├── LICENSE                     <- Open-source license
    ├── EXAM REPORT README.md        <- Main project README
   
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
### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Answer: Yes, we implemented several rules for code quality and formatting to maintain a clean and consistent codebase. We followed PEP8 guidelines using tools like Ruff for linting and code style enforcement. Our code is modular, with separate scripts for training, preprocessing, and serving, making it easier to test and maintain.

> We also used type hints in our functions to improve code clarity and help with debugging and static analysis. Additionally, we included docstrings to explain the purpose and expected inputs/outputs of functions, which improves code readability and helps other developers understand the logic faster.

> These practices are especially important in larger projects, where multiple people work on the same codebase over time. Consistent formatting, proper typing, and clear documentation reduce confusion, prevent bugs, and make onboarding new contributors easier. They also support better tooling and automated testing, which is critical for maintaining quality as the project grows.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer: Currently, all testing has been done manually. We have carefully tested the main functionalities, including data preprocessing, model training, and serving, to ensure everything works as expected. Manual testing helped us identify and fix issues during development, focusing on correctness and performance.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer: Since we relied on manual testing rather than automated tests, we don’t have a measurable code coverage percentage. Manual testing allowed us to verify key functionalities and expected behaviors through hands-on exploration.

> Even if we had automated tests with 100% coverage, that alone wouldn’t guarantee the code is completely error-free. Code coverage only shows which parts of the code were executed during testing, but it doesn’t ensure all edge cases are tested or that outputs are always correct.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer: Currently, our workflow uses a single branch called main. All development and updates are made directly on this branch. While this approach has worked for our current project size and team, using branches and pull requests can greatly improve version control and collaboration.

> Branches allow developers to work on features or fixes independently without affecting the main codebase. This reduces the risk of introducing bugs or conflicts. Pull requests then provide a formal way to review and discuss changes before merging them into the main branch, ensuring code quality and shared understanding. They also help maintain a clear history of changes and make it easier to roll back if needed.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer: Yes, we used DVC (Data Version Control) in our project to manage the raw_images/ folder, flower_labels.csv, and labels.csv. These files were essential for training and evaluating our model, and DVC helped us version them efficiently without storing large data directly in Git.

> Using DVC ensured that our data files remained in sync with our code, making it easier to reproduce experiments and track which data version was used for specific model runs. It also helped us avoid accidental overwrites or mismatches, especially when making updates to the dataset. Overall, DVC improved our workflow by bringing reliable version control to our data pipeline.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Answer: We did not use any continuous integration (CI) setup in this project. All testing and code checks were done manually, and there were no automated pipelines for tasks like unit testing, linting, or environment checks.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer: We configured our experiment using a ResNet-18 model with pretrained=True and trained it for 5 epochs. Since we used a pre-trained model, we didn’t require extensive configuration or hyperparameter tuning. Key parameters like the number of epochs were set directly in the training script instead of using external config files. Due to limited compute resources on the cloud platform (CPU-only for training and preprocessing), we kept the training duration short. Despite these constraints, the model achieved around 95% accuracy on validation, demonstrating that transfer learning can be highly effective even with minimal setup.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer: We ensured reproducibility of our experiments by using DVC for data versioning, logging to track experiment metrics, and Docker to maintain consistent environments. For each experiment, we version control the code and configuration with Git, track data versions with DVC, and containerize the setup using Docker. This way, anyone can reproduce results by checking out the correct Git commit, pulling the corresponding data version with DVC, and running the experiment inside the Docker container. Logging records all important metrics and parameters, allowing easy comparison and analysis of experiment outcomes.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer:
> ![wandb_runs1](figures/wandb_runs.jpg)
> ![wandb_runs2](figures/wandb_runs.jpg)
>
> We conducted two training runs of a ResNet18 model and tracked the results using Weights & Biases (W&B) for better experiment management and comparison. Both runs are marked as finished and include key metrics such as training/validation loss, accuracy, and training steps. In the first run, we achieved a validation accuracy of 95.91% and a validation loss of 0.3036. In the second run, we reached a validation accuracy of 95.66% with a validation loss of 0.3147. Both models were trained for 4 epochs across 224 global steps. We observed a steady decrease in training loss and a rise in accuracy, indicating effective learning and good generalization. These metrics are crucial for evaluating model performance and diagnosing potential overfitting or underfitting. By using W&B, we can easily compare runs, visualize trends, and ensure reproducibility in our MLOps pipeline, helping us make informed decisions when tuning or deploying models.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer: Docker played a crucial role in our MLOps pipeline by enabling reproducible and isolated environments for different stages of the workflow. We created separate >Docker images for preprocessing data, training the model, running the backend API, and serving the frontend UI:

> train_docker: Trains the ResNet18 model using the preprocessed data.
> https://github.com/praveentitusf/MLOps-Project-2025-Flower-Classification/blob/main/dockerfiles/preprocess.dockerfile

> By containerizing each component, we ensured consistency across development and deployment environments.

> Command to run docker image:
```
# Model training
docker run --rm train_docker
```

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer: Since we used PyTorch Lightning, a lot of boilerplate and error-prone code was abstracted away, which reduced the number of bugs during training.

> Interactive Debugging: We used the VSCode IDE’s built-in debugger to set breakpoints and inspect variables at runtime, which was particularly useful for catching logical errors.

> Print Statements: In simpler cases, we relied on print statements to quickly check tensor shapes, values, and control flow.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer:
> GCP Services Used in Our Project
> Compute Engine (VM Instances)
> We used virtual machines to run our training jobs, host APIs, and perform other development tasks. These VMs provided scalable compute resources with custom configurations.

> Cloud Storage
> Used to store large files such as datasets, model checkpoints, and output artifacts. It acted as a centralized and persistent storage layer accessible across services.

>Artifact Registry
>We used this to store and manage our Docker images. It allowed seamless integration with other GCP services like Cloud Run and Compute Engine, ensuring secure and versioned container deployments.

> Cloud Run
> Cloud Run was used to deploy our containerized backend API. It allowed us to run stateless HTTP services that automatically scale up or down based on traffic.

> Service Accounts
> Service accounts were configured to securely authenticate our containers and VMs when accessing other GCP services like Cloud Storage and Artifact Registry.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer: We used Google Compute Engine as the main compute resource for running various stages of our MLOps project, including model training, data preprocessing, and Dockerized applications. Compute Engine provided us with customizable virtual machines that we could scale based on the task requirements.

> Specifically, we used the n2-highmem-4 instance type, which offers 4 vCPUs and 32 GB of memory. This configuration gave us a good balance between compute power and memory, suitable for training deep learning models efficiently.

> We also configured Application Default Credentials (ADC) on the VM to securely access other GCP services such as Cloud Storage and Artifact Registry. This allowed the VM to authenticate without hardcoding credentials, making the setup secure and production-ready.

> Compute Engine was the backbone of our workflow, providing flexibility, control, and scalability during development and deployment.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:
> [gbucket](figures/gbucket.jpg)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:
> [artifact_repo](artifact_repo/gbucket.jpg)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer: In this project, instead of using an automated Cloud Build pipeline on Google Cloud Platform (GCP), we manually uploaded the built container images to the container registry. This means that the images were built in our development environment and then pushed directly to GCP’s Container Registry without triggering automated build history or build logs within the Cloud Build service.

> As a result, there is no automated Cloud Build history available to display. However, the container images are still available and can be verified in the Container Registry, showing the uploaded image versions and tags.

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Answer: We performed both preprocessing and model training on a Google Cloud Compute Engine instance. To maximize performance within our budget, we selected the best available VM suited for the task. Inside this VM, we installed Docker to containerize our workflows. Both the preprocessing and training Docker images were pulled directly from our Artifact Registry and run inside the Compute Engine VM. After preprocessing, the processed data was saved to a Google Cloud Storage bucket. Once training was complete, the resulting model checkpoint file (.ckpt) was also saved to the same bucket for persistent storage and later use.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Answer: Yes, we wrote an API for our model using FastAPI for the backend. The backend is responsible for handling image classification requests. It loads the model checkpoint and label mappings from Google Cloud Storage, processes the uploaded image, performs inference using the trained model, and returns the predicted class.

> We also built a separate frontend using Streamlit (for UI) and FastAPI (for routing). The frontend allows users to upload images and displays the model’s prediction after receiving the response from the backend API.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer: Yes, we wrote an API for our model using FastAPI and deployed it as a backend service. We created two separate Docker containers: one for the backend API and another for the frontend UI. The frontend, built with Streamlit, allows users to upload images and is deployed in its own container.

> The frontend sends the uploaded image to the backend API, which loads the model checkpoint and class labels from Google Cloud Storage, processes the image, performs inference, and returns the predicted class. The backend handles all model logic and serves predictions.

> Both containers are deployed independently on Google Cloud Run, enabling scalable, serverless hosting. This separation ensures modularity, easier maintenance, and the ability to scale each component independently.

> # Link to the deployed app: https://frontend-660622539098.europe-west1.run.app/

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Answer: We did not perform formal unit testing or load testing on our API. Instead, we relied on monitoring logs from Google Cloud Run for both the frontend and backend services. These logs provided real-time insights into the API’s behavior, including request handling, response times, and any errors encountered. By reviewing the logs, we were able to validate that the services were functioning correctly under expected usage and debug issues as they occurred. 

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer: We did not implement dedicated monitoring for our deployed model. However, we deployed both the frontend and backend services on Google Cloud Run, which provides built-in logging and basic metrics such as request count, error rates, and response latency. These logs allowed us to observe and debug the application during development and testing.

> Proper monitoring would significantly improve the longevity and reliability of our application. With a monitoring setup, we could track key metrics such as model performance over time, latency, API usage patterns, and potential data drift. This would help identify performance degradation, detect anomalies, and trigger alerts for unexpected behavior.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Answer:
> 
```
| Service             | Cost (€) |
|---------------------|----------|
| Compute Engine      | €8.91    |
| Cloud Storage       | €2.60    |
| Cloud Run           | €0.73    |
| Networking          | €0.11    |
| VM Manager          | €0.10    |
| Artifact Registry   | €0.01    |
| Cloud Run Functions | €0.00    |
| **Total**           | €12.46    |
```
> Working in the cloud was a valuable learning experience. It allowed us to quickly deploy, test, and scale different components of our application without worrying about managing physical infrastructure. Services like Compute Engine and Cloud Run gave us flexibility in how we handled training, inference, and deployment.

> The pay-as-you-go model kept costs manageable, especially since we were able to use free tier services for some components. However, Compute Engine was the most expensive part of our setup due to its heavy use during model training.

> Overall, the cloud made development faster, deployment smoother, and team collaboration easier. For future projects, we’d aim to optimize usage further and explore cost-saving features like sustained use discounts or committed use contracts.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Answer: We did not implement any extra features beyond what was covered in the course material. The components and tools provided in the course, such as FastAPI for serving the model, Docker for containerization, Google Cloud services for deployment, and Streamlit for the frontend were sufficient for building a complete and functional MLOps pipeline. We focused on understanding and applying these core concepts effectively rather than adding additional complexity.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> [workflow](figures/workflow.jpg)
> 
> Answer: The starting point of our system is the local development environment, where we build Docker images for both the model training pipeline and the application deployment pipeline. For the training pipeline, we create Docker containers that include preprocessing and model training scripts. Similarly, for deployment, we build Docker images containing the backend service (implemented using FastAPI) and the frontend interface (using Streamlit).

> Once the Docker images are built locally, they are pushed to the Google Artifact Registry, which acts as a centralized repository for container images, enabling easy version control and deployment. The training pipeline images are then deployed to a Google Cloud Compute Engine (GCE) virtual machine, where the preprocessing and model training steps are executed. After the model is trained, the resulting model artifacts are saved to Google Cloud Storage, providing persistent and scalable storage.

> For the deployment pipeline, the backend and frontend Docker images are deployed to Google Cloud Run, a fully managed serverless platform that handles container execution and scaling automatically. The frontend presents a user interface where users can upload images. These images are sent to the backend service, which loads the trained model from Google Cloud Storage to generate predictions. Finally, the prediction results are returned and displayed on the frontend UI.

> This architecture separates concerns by isolating model training from deployment, leveraging managed cloud services for scalability, and ensuring a smooth workflow from data preprocessing to real-time prediction. The use of Docker containers throughout guarantees reproducibility and consistent environments across local development and cloud infrastructure.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer: One of the biggest challenges we faced in this project was setting up the cloud infrastructure on Google Cloud Platform (GCP). Specifically, configuring service account roles and setting up Application Default Credentials (ADC) for the Virtual Machine (VM) was time-consuming and required careful attention to permission scopes. The documentation often lacked clarity, and we had to troubleshoot errors that stemmed from missing IAM roles or improperly configured ADCs.

> Another major challenge was identifying the right VM configuration for training and preprocessing tasks. Due to quota limitations on our trial account, we had to test different regions and machine types to find a setup that met our resource requirements and was available for deployment. This process was iterative and required us to adapt our Docker containers to ensure compatibility across environments.

> We also spent a significant amount of time developing and containerizing both the backend and frontend components. The backend, built using FastAPI, handled prediction requests, model loading, and logging. The frontend, created with Streamlit, needed to be intuitive and responsive, while integrating smoothly with the backend API. Deploying these to Cloud Run involved learning how to write efficient Dockerfiles, manage environment variables, and troubleshoot deployment errors.

> Integrating the frontend and backend presented additional challenges, such as CORS issues, asynchronous request handling, and ensuring real-time responsiveness. Debugging these issues was difficult due to scattered documentation and platform-specific behavior.

> Overall, we spent the most time on creating and integrating the backend and frontend systems. We overcame these obstacles by dividing tasks, sharing knowledge, consulting forums, and iteratively testing each module. This hands-on experience significantly improved our understanding of deploying full-stack machine learning systems on the cloud.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Answer:
> Ali Najibpour Nashi:
>Ali developed the training and preprocessing scripts and located the required datasets. He ensured the data was properly prepared for model training. His work laid the foundation for the entire machine learning pipeline.

> Dileep Vemuri:
Dileep set up and configured the cloud virtual machine, including installing Docker inside the VM. He managed service accounts and created cloud storage buckets to handle data securely. Additionally, he integrated Weights & Biases (WandB) for experiment tracking.

> Praveen Titus Francis:
Titus containerized the training, preprocessing, backend, and frontend components using Docker. He developed the backend and frontend services and deployed them on Google Cloud Run. His work enabled smooth deployment and access to the full application.

# Garbage Classification - End-to-End ML Pipeline

This project demonstrates the complete end-to-end Machine Learning cycle, from data acquisition and preprocessing to model training, deployment, and monitoring. The use case is image classification for garbage items (cardboard, glass, metal, paper, plastic, trash) using a fine-tuned MobileNetV2 model.

## Links

- **GitHub Repository**: [Insert GitHub Link Here]
- **Video Demo (YouTube)**: [Insert YouTube Link Here]
- **Live App URL (Render)**: [Insert Render URL Here]

## Project Description

This pipeline fulfills the following requirements:
- **Offline Model Creation**: Jupyter notebook containing EDA, preprocessing, and training code using a pre-trained MobileNetV2 architecture.
- **RESTful API**: A FastAPI service for model predictions, fetching uptime and evaluation metrics, and triggering background model retraining.
- **Interactive UI**: A Streamlit dashboard that visualizes data features, allows users to upload an image for prediction, and provides a drag-and-drop interface to upload bulk data for retraining.
- **Retraining Trigger**: The UI and API support receiving a ZIP file of new images, merging them with the existing raw dataset, re-splitting the train/test sets, and triggering a retraining cycle. A guard logic ensures the model is only updated if it doesn't degrade.
- **Load Testing**: Locust is used to simulate a flood of traffic against the API, which can be scaled out using Nginx and Docker Compose.

## Directory Structure
- `notebook/`: Jupyter notebook for initial EDA and training.
- `src/`: Reusable Python modules for preprocessing, model definition, and prediction logic.
- `api/`: FastAPI application and Dockerfile.
- `ui/`: Streamlit application and Dockerfile.
- `locust/`: Load testing scripts.
- `nginx/`: Reverse proxy configuration for load balancing.
- `render.yaml`: Blueprint for cloud deployment.

## Setup Instructions

### 1. Initial Setup and Model Generation
Before running the API and UI, you need to generate the initial model and visualizations:
1. Open `notebook/garbage_classification.ipynb` in Jupyter or VS Code.
2. Run all cells. This will download the dataset via Kagglehub, train the model, save visualizations to `data/visualizations/`, and save the model to `models/garbage_classifier_v1.h5`.

### 2. Running Locally with Docker Compose
The project uses Docker Compose to orchestrate the UI, Nginx load balancer, and scalable API instances.

```bash
# Start the services (1 API container)
docker-compose up --build -d

# Scale the API to 2 containers
docker-compose up --scale api=2 -d

# Scale the API to 4 containers
docker-compose up --scale api=4 -d
```
The Streamlit UI will be available at `http://localhost:8501`.

### 3. Deploying to Render
1. Push this repository to GitHub.
2. Connect your GitHub account to Render.
3. Click "New" -> "Blueprint" and select the repository.
4. Render will use the `render.yaml` file to automatically spin up both the API and the UI.
5. Once deployed, the UI will automatically connect to the API via the environment variables defined in the Blueprint.

## Flood Request Simulation (Locust)

We used Locust to simulate a flood of requests to the `/predict` endpoint under different scaling configurations.

*Note: Replace the values below with the actual results from running Locust locally against the different scaled API containers.*

| Containers | Requests/sec (RPS) | Average Latency (ms) | Max Latency (ms) |
|------------|--------------------|----------------------|------------------|
| 1 API      | [Record Value]     | [Record Value]       | [Record Value]   |
| 2 APIs     | [Record Value]     | [Record Value]       | [Record Value]   |
| 4 APIs     | [Record Value]     | [Record Value]       | [Record Value]   |

To run the load test yourself:
```bash
pip install locust Pillow
locust -f locust/locustfile.py --host=http://localhost:8000
```
Then navigate to `http://localhost:8089` to start the test.

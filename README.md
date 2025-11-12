# Week 7 - MLOps Project

FastAPI Iris prediction service with Kubernetes deployment and CI/CD automation.

## Project Structure

```
week7/
├── main.py                    # Entry point for the FastAPI Iris prediction service
├── Dockerfile                 # Docker configuration for containerizing the FastAPI service
├── requirements.txt           # List of dependencies for the project
├── wrk_script.lua            # wrk example load script
├── .gitignore                # Specifies files to be ignored by git
│
├── k8s/                      # Kubernetes configurations
│   ├── deployment.yaml       # Kubernetes deployment configuration
│   ├── service.yaml          # Kubernetes service configuration
│   └── hpa.yaml              # HorizontalPodAutoscaler configuration
│
├── .github/
│   └── workflows/            # GitHub Actions workflows
│       ├── ci-main.yml       # CI/CD workflow for main branch
│
├── data/
│   └── iris.csv              # Iris dataset
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation script
│
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_data_validation.py  # Data validation tests
    └── test_evaluation.py       # Model evaluation tests
```

## Features

- **FastAPI Service**: REST API for Iris flower classification
- **Containerization**: Docker support for easy deployment
- **Kubernetes**: Full K8s deployment with services and autoscaling
- **CI/CD**: Automated workflows for different branches
- **Load Testing**: Integration with wrk for performance testing
- **Testing**: Comprehensive test suite for data validation and model evaluation

## Getting Started

### Prerequisites

- Python 3.x
- Docker
- Kubernetes cluster (for K8s deployment)
- kubectl (for K8s management)

### Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI service locally:
   ```bash
   python main.py
   ```

### Docker Deployment

Build and run the Docker container:

```bash
docker build -t iris-prediction-service .
docker run -p 8000:8000 iris-prediction-service
```

### Kubernetes Deployment

Deploy to Kubernetes:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## CI/CD Workflows

- **ci-main.yml**: ci/cd workflow for main branch, which includes the stress tests

## Testing

Run the test suite:

```bash
pytest tests/
```

# Industrial Predictive Maintenance Platform 🏗️🤖

This is a comprehensive, end-to-end Machine Learning project focused on **Industrial AI** and **Predictive Maintenance**. It simulates a real-world monitoring system for industrial machines, providing failure predictions, anomaly detection, and sensor forecasting.

## 🚀 Project Overview

The goal of this platform is to transform raw sensor data into actionable insights to prevent machine downtime. It covers the entire ML lifecycle: from data exploration to production deployment with Docker.

### Key Features:
- **ML Failure Prediction**: Supervised learning to predict machine failure risk.
- **Unsupervised Anomaly Detection**: Isolation Forest model to detect abnormal patterns.
- **Time-Series Forecasting**: ARIMA model to predict future sensor values (Temperature).
- **Production Pipeline**: Automated preprocessing and model serving via Scikit-Learn Pipelines.
- **FastAPI Model Serving**: Real-time REST API for live inference.
- **Live Monitoring Simulation**: Script to simulate real-time sensor streams and alerts.
- **Interactive Dashboard**: Streamlit-based UI for data visualization and monitoring.
- **Containerization**: Docker & Docker-Compose for portable deployment.

---

## 🛠️ Tech Stack
- **Languages**: Python 3.11+
- **ML Frameworks**: Scikit-Learn, Statsmodels
- **Data Handling**: Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Deployment**: FastAPI, Uvicorn, Docker, Docker-Compose

---

## 📂 Project Structure
```text
industrial-ml-learning/
│
├── datasets/            # Raw industrial datasets (AI4I 2020)
├── models/              # Trained models and pipelines (.pkl)
├── notebooks/           # Step-by-step learning notebooks (01-08)
├── src/
│   ├── data/            # Data processing scripts
│   ├── features/        # Feature engineering logic
│   ├── inference/       # FastAPI app for model serving
│   ├── monitoring/      # Live sensor stream simulation
│   └── dashboard/       # Streamlit UI dashboard
│
├── Dockerfile           # Blueprint for the API container
├── docker-compose.yml   # Multi-container management
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚦 How to Run

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/kingkenthealpha/industrial-ml-learning.git
cd industrial-ml-learning

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API
```bash
uvicorn src.inference.app:app --reload
# Access Swagger UI at http://127.0.0.1:8000/docs
```

### 3. Run the Dashboard
```bash
streamlit run src.dashboard.dashboard:app
```

### 4. Run Docker (Recommended)
```bash
docker-compose up --build
```

---

## 📈 Learning Journey
This project was built through a 10-phase micro-step roadmap:
1. Environment Setup & Python Base
2. Pandas Mastery & Real Data Handling
3. Modeling & Training (Random Forest)
4. Industrial ML Evaluation (Recall, Precision)
5. Advanced Feature Engineering (Rolling Stats)
6. Time-Series Forecasting (ARIMA)
7. Unsupervised Anomaly Detection (Isolation Forest)
8. Production Pipelines (Sklearn Pipeline)
9. FastAPI Deployment
10. Docker & Containerization

---

## 👤 Author
**Dharmik Kothari**
Industrial ML Engineer in training.

---
*Built with passion for Industrial AI and Machine Learning Excellence.*

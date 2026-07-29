# ✈️ PredictX AI: Predictive Maintenance Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-15C4CE?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)](https://render.com/)

**PredictX AI** is a full-stack, machine learning web application designed to predict the **Remaining Useful Life (RUL)** of turbofan engines. By leveraging complex sensor data, it identifies failing components in real-time to prevent catastrophic equipment failures and optimize industrial maintenance schedules.

### 🚀 [View the Live Application Here](https://predictx-frontend-g86o.onrender.com)

---

## 🏗️ System Architecture

This project is engineered as a **Monolithic Docker Container**, designed for zero-latency internal communication and seamless cloud deployment.

*   **Frontend (Streamlit):** Handles the interactive UI, Plotly charts, and file uploads. Runs on the exposed external port (`8501`).
*   **Backend (FastAPI):** Exposes a REST API for model inference. Runs internally in the background (`Port 8000`).
*   **Network:** The frontend and backend share `localhost` inside the unified container, completely eliminating network latency during prediction requests.
*   **Deployment:** Containerized via Docker and continuously deployed on Render.

---

## ✨ Key Features

*   **🧠 Machine Learning Inference:** Utilizes an optimized **XGBoost Regressor** to accurately predict engine failure cycles based on historical sensor data.
*   **🔍 Real-Time Explainability (SHAP):** Generates SHapley Additive exPlanations (SHAP) to show exactly *which* sensors (e.g., temperature, static pressure) are driving the failure prediction.
*   **🚦 Automated Risk Assessment:** Dynamically calculates an "Engine Health Score" (0-100) and categorizes engines into **Healthy**, **Medium Risk**, or **High Risk** to trigger maintenance alerts.
*   **📈 Historical Sensor Tracking:** Interactive Plotly line charts allow users to isolate and track specific sensor degradation over the lifecycle of the engine.
*   **⚡ 1-Click Interviewer Demo:** Features a built-in data loader allowing users to instantly test the application using the pre-configured NASA CMAPSS `FD001` test dataset without needing to download files locally.
*   **📥 Exportable Reports:** Generate and download complete JSON diagnostic reports for individual engines.

---

## 📊 About the Dataset (NASA CMAPSS)

The model is trained on the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** benchmark dataset, specifically the **FD001** subset. 

*   **Environment:** Simulates engines operating under a single condition (sea level) experiencing a single fault mode (HPC degradation).
*   **Input:** 21 distinct sensor measurements recorded across multiple operational cycles.
*   **Objective:** Predict the exact cycle at which the engine will cross the threshold of failure.

---

## 🛠️ Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Machine Learning** | Python, XGBoost, Scikit-Learn, Pandas, NumPy |
| **Model Explainability** | SHAP (SHapley Additive exPlanations) |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit, Plotly Express |
| **Infrastructure & DevOps** | Docker, Bash scripting, Render, Git |

---

## 💻 Local Installation & Setup

If you wish to run this project locally, ensure you have [Docker](https://www.docker.com/) installed.

### Option 1: Run via Docker (Recommended)
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Princegupta-78/PredictX-AI.git](https://github.com/Princegupta-78/PredictX-AI.git)
   cd PredictX-AI

# 🚀 SBA Commercial Credit Decision Engine | Real-Time MLOps Pipeline

[![CI/CD Pipeline](https://github.com/ashwinrcs/sba-credit-decision-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/ashwinrcs/sba-credit-decision-api/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.2-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg?logo=googlecloud)](https://cloud.google.com/run)

An end-to-end Machine Learning Operations (MLOps) pipeline for real-time SME credit risk underwriting. This system predicts the probability of default for US Small Business Administration (SBA) 7(a) commercial loans.

🔴 **Live API Endpoint (Swagger UI):** [https://sba-decision-api-oh4e6zcfyq-el.a.run.app/docs](https://sba-decision-api-oh4e6zcfyq-el.a.run.app/docs) 

---

## 🏗️ System Architecture

This project strictly separates model training from model serving, ensuring zero training-serving skew and highly scalable inference.

1. **Model Pipeline:** A custom `scikit-learn` ColumnTransformer handles missing value imputation, one-hot encoding, and scaling *inside* the serialized `.joblib` artifact.
2. **Inference API:** A decoupled FastAPI service validates incoming JSON payloads using strict Pydantic schemas. 
3. **Containerization:** The inference engine is packaged in an optimized `python:3.12-slim` Docker container.
4. **CI/CD Automation:** GitHub Actions automatically rebuilds and pushes the Docker container to Docker Hub on every commit to `main`.
5. **Cloud Deployment:** The container is hosted serverless-ly on Google Cloud Run (Mumbai region) for ultra-low latency inference.

---

## ⚡ Key Engineering Features

*   **Zero Training-Serving Skew:** By serializing the entire `sklearn.pipeline.Pipeline`, the FastAPI endpoint requires zero Pandas data wrangling. The model artifact handles raw JSON conversion internally.
*   **Strict Data Validation:** Pydantic models reject malformed requests (e.g., negative loan amounts, invalid NAICS codes) before they ever reach the ML model, preventing server crashes.
*   **Automated CI/CD:** Fully automated GitHub Actions workflow (`deploy.yml`) for continuous delivery.
*   **Data Leakage Prevention:** Built without using post-origination variables (like `DisbursementDate` or `ChargeOffAmount`) to reflect a true production underwriting environment.

---

## 💻 Test the API (Live Cloud Endpoint)

You can send a POST request to the live Cloud Run endpoint using `cURL` or Python. 

### High-Risk "Toxic Loan" Payload Example
*A 1-year revolving line of credit for a brand-new restaurant in an urban area with no real estate backing.*

```bash
curl -X 'POST' \
  '[https://sba-decision-api-oh4e6zcfyq-el.a.run.app/predict](https://sba-decision-api-oh4e6zcfyq-el.a.run.app/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Term": 12,
  "NoEmp": 1,
  "CreateJob": 0,
  "RetainedJob": 1,
  "GrAppv": 25000.0,
  "Guarantee_Ratio": 0.5,
  "NAICS_Sector": "72",
  "NewExist": "2",
  "UrbanRural": "1",
  "IsFranchise": "0",
  "RealEstate": "0",
  "RevLineCr": "Y",
  "LowDoc": "N"
}'
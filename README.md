# Federated Health Risk Prediction System

A privacy-preserving, federated learning system for health risk prediction using real-world open datasets. Simulates Apple Watch/Fitbit signals for research and prototyping.

## Features
- Federated Learning (Flower + PyTorch)
- Real-world data integration (WESAD, Fitbit, MIMIC-IV)
- Personalization layer (per-user fine-tuning, SHAP explanations)
- Streamlit dashboard (training stats, risk scores, feature importances)
- Docker Compose orchestration
- CI with GitHub Actions

## Architecture

```
[Server] <--> [Clients 1..N] <--> [Dashboard]
```

## File Structure
- backend/
  - server.py
  - model.py
  - utils/
- clients/
  - client_1/train.py
  - client_N/train.py
- data/
  - raw/
  - processed/
  - generate_dataset.py
- dashboard/
  - app.py
  - visualizations.py
- Dockerfile
- docker-compose.yml
- requirements.txt
- .github/workflows/ci.yml

## How to Run

1. Generate placeholder data:
   ```bash
   python data/generate_dataset.py
   ```
2. Start with Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. Access dashboard at http://localhost:8501

## Datasets Used
- WESAD: https://archive.ics.uci.edu/ml/datasets/WESAD
- Fitbit Public Dataset: https://www.fitabase.com/databank/
- MIMIC-IV Waveform: https://physionet.org/content/mimiciv/

## Screenshots
- [Placeholder for dashboard screenshots]

## Citations
- [WESAD Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD)
- [Fitbit Public Dataset](https://www.fitabase.com/databank/)
- [MIMIC-IV Waveform](https://physionet.org/content/mimiciv/) 
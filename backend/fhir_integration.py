import requests
import pandas as pd
import os
import argparse
import numpy as np
from collections import defaultdict

FHIR_BASE_URL = os.environ.get("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
PROCESSED_DIR = "data/processed"

# Mapping from FHIR Observation type to model feature
FHIR_TO_MODEL = {
    'Heart rate': 'heart_rate',
    'Heart Rate': 'heart_rate',
    'Pulse': 'heart_rate',
    'Step count': 'steps',
    'Steps': 'steps',
    'Sleep duration': 'sleep',
    'Total sleep time': 'sleep',
    'Stress level': 'stress_level',
    'Pain severity - 0-10 verbal numeric rating [Score] - Reported': 'stress_level',
    'Anxiety score': 'stress_level',
    'Oxygen saturation': 'spO2',
    'SpO2': 'spO2',
}
MODEL_FEATURES = ['heart_rate', 'steps', 'sleep', 'stress_level', 'spO2']

# Fetch FHIR Patient resources
def fetch_patients():
    resp = requests.get(f"{FHIR_BASE_URL}/Patient")
    resp.raise_for_status()
    return resp.json().get('entry', [])

# Fetch Observations for a patient
def fetch_observations(patient_id):
    resp = requests.get(f"{FHIR_BASE_URL}/Observation?subject=Patient/{patient_id}")
    resp.raise_for_status()
    return resp.json().get('entry', [])

# Parse FHIR Observation to dict
def parse_observation(obs):
    resource = obs['resource']
    code = resource.get('code', {}).get('text', '')
    value = resource.get('valueQuantity', {}).get('value', None)
    date = resource.get('effectiveDateTime', '')
    if date:
        date = date[:10]  # Use only YYYY-MM-DD
    return {'date': date, 'type': code, 'value': value}

# Convert FHIR data to time series per-user CSVs
def fhir_to_timeseries_csv():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    patients = fetch_patients()
    for i, entry in enumerate(patients):
        patient = entry['resource']
        pid = patient['id']
        obs_entries = fetch_observations(pid)
        obs_data = [parse_observation(obs) for obs in obs_entries if parse_observation(obs)['date']]
        # Group by date
        by_date = defaultdict(dict)
        for obs in obs_data:
            model_feat = FHIR_TO_MODEL.get(obs['type'])
            if model_feat and obs['value'] is not None:
                by_date[obs['date']][model_feat] = obs['value']
        # Build time series rows
        rows = []
        for date, feats in by_date.items():
            row = {'date': date}
            for f in MODEL_FEATURES:
                row[f] = feats.get(f, np.nan)
            # Synthesize a label (random for demo)
            row['cardiovascular_risk'] = float(np.random.rand() > 0.5)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('date')
            df.to_csv(os.path.join(PROCESSED_DIR, f"user_{i+1}.csv"), index=False)
            print(f"Wrote user_{i+1}.csv for Patient {pid} with {len(df)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FHIR data and generate per-user time series CSVs.")
    parser.add_argument('--to-csv', action='store_true', help='Fetch FHIR data and write per-user time series CSVs')
    args = parser.parse_args()
    if args.to_csv:
        fhir_to_timeseries_csv() 
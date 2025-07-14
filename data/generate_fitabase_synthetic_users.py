import os
import numpy as np
import pandas as pd

# Paths to raw data
HR_PATH = "fitabaseexampledata/HeartRate/ID 1003_heartrate_1min_20171001_20171007.csv"
STEPS_PATH = "fitabaseexampledata/Steps/ID 1003_minuteStepsNarrow_20171001_20171007.csv"
SLEEP_PATH = "fitabaseexampledata/SleepClassic/ID 1003_minuteSleep_20171001_20171007.csv"
CAL_PATH = "fitabaseexampledata/Calories/ID 1003_minuteCaloriesNarrow_20171001_20171007.csv"
INTENSITY_PATH = "fitabaseexampledata/Intensity/ID 1003_minuteIntensitiesNarrow_20171001_20171007.csv"

PROCESSED_DIR = "data/processed"
N_SYNTHETIC_USERS = 5

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_merge():
    # Load data
    hr = pd.read_csv(HR_PATH)
    steps = pd.read_csv(STEPS_PATH)
    sleep = pd.read_csv(SLEEP_PATH)
    cal = pd.read_csv(CAL_PATH)
    intensity = pd.read_csv(INTENSITY_PATH)

    # Standardize timestamp columns
    hr['timestamp'] = pd.to_datetime(hr['Time'])
    steps['timestamp'] = pd.to_datetime(steps['ActivityMinute'])
    sleep['timestamp'] = pd.to_datetime(sleep['date'])
    cal['timestamp'] = pd.to_datetime(cal['ActivityMinute'])
    intensity['timestamp'] = pd.to_datetime(intensity['ActivityMinute'])

    # Merge on timestamp (inner join for aligned data)
    df = hr[['timestamp', 'Value']].rename(columns={'Value': 'heart_rate'})
    df = df.merge(steps[['timestamp', 'Steps']], on='timestamp', how='inner')
    df = df.merge(sleep[['timestamp', 'value']], on='timestamp', how='left').rename(columns={'value': 'sleep'})
    df = df.merge(cal[['timestamp', 'Calories']], on='timestamp', how='left')
    df = df.merge(intensity[['timestamp', 'Intensity']], on='timestamp', how='left')

    # Fill missing values
    df['sleep'] = df['sleep'].fillna(0)
    df['Calories'] = df['Calories'].fillna(0)
    df['Intensity'] = df['Intensity'].fillna(0)

    # Feature engineering
    df['steps'] = df['Steps']
    df['stress_level'] = df['Intensity'] / (df['Intensity'].max() + 1e-8)  # proxy
    df['spO2'] = np.random.uniform(95, 100, len(df))  # simulate spO2
    df['cardiovascular_risk'] = (df['heart_rate'] > 90).astype(float) * 0.7 + (df['stress_level'] > 0.5).astype(float) * 0.3
    df['cardiovascular_risk'] = df['cardiovascular_risk'] + np.random.normal(0, 0.05, len(df))
    df['cardiovascular_risk'] = df['cardiovascular_risk'].clip(0, 1)

    # Final columns
    df = df[['timestamp', 'heart_rate', 'steps', 'sleep', 'stress_level', 'spO2', 'cardiovascular_risk']]
    df = df.rename(columns={'timestamp': 'date'})
    return df

def make_synthetic_users(df, n_users=5):
    for i in range(1, n_users+1):
        user_df = df.copy()
        # Perturb features for each synthetic user
        user_df['heart_rate'] += np.random.normal(0, 2, len(user_df))
        user_df['steps'] += np.random.normal(0, 5, len(user_df))
        user_df['sleep'] += np.random.normal(0, 0.1, len(user_df))
        user_df['stress_level'] += np.random.normal(0, 0.05, len(user_df))
        user_df['spO2'] += np.random.normal(0, 0.2, len(user_df))
        user_df['cardiovascular_risk'] += np.random.normal(0, 0.05, len(user_df))
        # Robust clipping: only clip columns that exist
        clip_min = {'heart_rate': 40, 'steps': 0, 'sleep': 0, 'stress_level': 0, 'spO2': 90, 'cardiovascular_risk': 0}
        clip_max = {'heart_rate': 200, 'steps': 500, 'sleep': 12, 'stress_level': 1, 'spO2': 100, 'cardiovascular_risk': 1}
        existing_cols = [col for col in user_df.columns if col in clip_min]
        user_df[existing_cols] = user_df[existing_cols].clip(
            lower=[clip_min[c] for c in existing_cols],
            upper=[clip_max[c] for c in existing_cols],
            axis=1
        )
        user_df.to_csv(f"{PROCESSED_DIR}/user_{i}.csv", index=False)
        print(f"[Data] Saved synthetic user {i} data to {PROCESSED_DIR}/user_{i}.csv")

if __name__ == "__main__":
    df = load_and_merge()
    make_synthetic_users(df, N_SYNTHETIC_USERS)
    print("[Data] All synthetic user data generated.") 
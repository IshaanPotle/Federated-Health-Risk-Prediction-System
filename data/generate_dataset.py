"""
Stub for generating and partitioning dataset by user for federated simulation.
"""
import os
import numpy as np
import pandas as pd

def generate_placeholder_data(num_users=5, num_samples=1000):
    os.makedirs("data/processed", exist_ok=True)
    for user_id in range(1, num_users+1):
        df = pd.DataFrame({
            "heart_rate": np.random.randint(60, 100, num_samples),
            "steps": np.random.randint(0, 200, num_samples),
            "sleep": np.random.uniform(4, 9, num_samples),
            "stress_level": np.random.uniform(0, 1, num_samples),
            "spO2": np.random.uniform(95, 100, num_samples),
            "cardiovascular_risk": np.random.uniform(0, 1, num_samples),
        })
        df.to_csv(f"data/processed/user_{user_id}.csv", index=False)
    print("[Data] Placeholder user data generated.")

if __name__ == "__main__":
    generate_placeholder_data() 
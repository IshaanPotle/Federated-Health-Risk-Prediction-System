import os
import random
import re

PROCESSED_DIR = 'data/processed'
CLIENTS_DIR = 'clients'
NUM_CLIENTS = 5

# List all user CSVs in data/processed/
user_csvs = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.csv') and f.startswith('user_')])

# Assign each client a user CSV (round-robin)
assignments = {}
for i in range(NUM_CLIENTS):
    client_name = f'client_{i+1}'
    csv_file = user_csvs[i % len(user_csvs)]
    assignments[client_name] = csv_file

# Update data_path in each client's train.py
for client, csv_file in assignments.items():
    train_py = os.path.join(CLIENTS_DIR, client, 'train.py')
    if not os.path.exists(train_py):
        print(f"{train_py} not found, skipping.")
        continue
    with open(train_py, 'r') as f:
        lines = f.readlines()
    new_lines = []
    replaced = False
    for line in lines:
        if re.match(r'\s*data_path\s*=.*', line):
            new_lines.append(f'data_path = "{PROCESSED_DIR}/{csv_file}"\n')
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        # Insert at the top if not found
        new_lines.insert(0, f'data_path = "{PROCESSED_DIR}/{csv_file}"\n')
    with open(train_py, 'w') as f:
        f.writelines(new_lines)
    print(f"Assigned {csv_file} to {client}/train.py")

print("\nAssignment summary:")
for client, csv_file in assignments.items():
    print(f"{client}: {csv_file}") 
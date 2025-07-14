# backend/server.py
"""
Federated Learning Server (Flower-based)
Orchestrates federated rounds and aggregates client updates.
"""
import flwr as fl

def main():
    print("[Server] Starting federated server...")
    # Use FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # All clients participate
        min_fit_clients=5,
        min_available_clients=5,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

if __name__ == "__main__":
    main() 
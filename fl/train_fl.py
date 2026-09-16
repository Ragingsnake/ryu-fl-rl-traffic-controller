import argparse
import flwr as fl
import os
from fl.client import TrafficClient, load_data
from fl.model import create_model
from fl.server import build_strategy

def main():
    parser = argparse.ArgumentParser(description="Train Federated Learning Model for Traffic Prediction")
    parser.add_argument("--rounds", type=int, default=100, help="Number of FL rounds to train")
    parser.add_argument("--data_dir", type=str, default="data/preprocessed", help="Directory containing preprocessed data")
    parser.add_argument("--model_save_path", type=str, default="models/fl_global_model.pt", help="Path to save the global model")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    import numpy as np

    # Determine MAX_F_IN dynamically across domain data
    max_f = 24
    for did in [0, 1, 2]:
        p = os.path.join(args.data_dir, f"domain_{did}_train_X.npy")
        if os.path.exists(p):
            max_f = max(max_f, np.load(p, mmap_mode='r').shape[2])
    MAX_F_IN = max_f
    
    def client_fn(cid: str) -> fl.client.Client:
        domain_id = int(cid)
        train_loader, val_loader, f_in = load_data(
            domain_id, data_dir=args.data_dir, batch_size=32, target_f_in=MAX_F_IN
        )
        model = create_model(f_in=f_in)
        
        return TrafficClient(domain_id=domain_id, model=model, train_loader=train_loader, val_loader=val_loader).to_client()
        
    strategy = build_strategy(
        num_rounds=args.rounds, 
        num_clients=3,
        model_save_path=args.model_save_path,
        f_in=MAX_F_IN
    )
    
    print(f"Starting Flower simulation for {args.rounds} rounds...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    print("FL Training complete!")

if __name__ == "__main__":
    main()

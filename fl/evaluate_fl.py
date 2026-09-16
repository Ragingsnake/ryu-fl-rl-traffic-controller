from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl.model import create_model, predict


def compute_metrics(y_true, y_pred):
    # Shape: (N, H, F_in) -> reshape to (N*H, F_in)
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    mae = np.mean(np.abs(y_true_flat - y_pred_flat), axis=0)
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2, axis=0))
    mape = np.mean(np.abs(y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-6), axis=0) * 100

    # R2 score manually
    r2 = []
    for i in range(y_true_flat.shape[1]):
        ss_res = np.sum((y_true_flat[:, i] - y_pred_flat[:, i]) ** 2)
        ss_tot = np.sum((y_true_flat[:, i] - np.mean(y_true_flat[:, i])) ** 2)
        r2_val = 1 - (ss_res / (ss_tot + 1e-10))
        r2.append(r2_val)
    r2 = np.array(r2)

    return mae, rmse, mape, r2


def main():
    print("Starting FL Prediction Evaluation...")
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data" / "preprocessed"
    model_path = base_path / "models" / "fl_global_model.pt"
    results_dir = base_path / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    global_state_dict = None
    if model_path.exists():
        global_state_dict = torch.load(model_path, map_location="cpu")
    else:
        print(f"Warning: {model_path} not found. Ensure the FL model is trained.")
        return

    MAX_F_IN = 24
    if global_state_dict is not None and "lstm.weight_ih_l0" in global_state_dict:
        MAX_F_IN = global_state_dict["lstm.weight_ih_l0"].shape[1]
    results_list = []

    for domain_id in [0, 1, 2]:
        test_x_path = data_dir / f"domain_{domain_id}_test_X.npy"
        test_y_path = data_dir / f"domain_{domain_id}_test_Y.npy"

        if not test_x_path.exists() or not test_y_path.exists():
            print(f"Test data for domain {domain_id} not found. Skipping.")
            continue

        X_test = np.load(test_x_path)
        Y_test = np.load(test_y_path)
        f_in = X_test.shape[-1]

        # Try to load model with f_in
        model = create_model(f_in=f_in)
        padded = False
        try:
            model.load_state_dict(global_state_dict)
        except RuntimeError:
            # Dimension mismatch, use MAX_F_IN and pad
            print(f"Shape mismatch for domain {domain_id}. Padding to MAX_F_IN={MAX_F_IN}.")
            model = create_model(f_in=MAX_F_IN)
            model.load_state_dict(global_state_dict)
            padded = True

        model.eval()

        # Run inference in batches to avoid memory issues if data is large
        batch_size = 64
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_x = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32)
                if padded and batch_x.shape[-1] < MAX_F_IN:
                    pad_size = MAX_F_IN - batch_x.shape[-1]
                    batch_x = torch.nn.functional.pad(batch_x, (0, pad_size))

                batch_pred = predict(model, batch_x)
                if padded and f_in < MAX_F_IN:
                    batch_pred = batch_pred[:, :, :f_in]

                preds.append(batch_pred.numpy())

        Y_pred = np.concatenate(preds, axis=0)

        mae, rmse, mape, r2 = compute_metrics(Y_test, Y_pred)

        print(f"\n--- Domain {domain_id} Metrics ---")
        print(f"Aggregated MAE:  {np.mean(mae):.4f}")
        print(f"Aggregated RMSE: {np.mean(rmse):.4f}")
        print(f"Aggregated MAPE: {np.mean(mape):.4f}%")
        print(f"Aggregated R2:   {np.mean(r2):.4f}")

        for i in range(f_in):
            results_list.append(
                {"Domain": domain_id, "Link_Idx": i, "MAE": mae[i], "RMSE": rmse[i], "MAPE": mape[i], "R2": r2[i]}
            )

    if results_list:
        df = pd.DataFrame(results_list)
        out_csv = results_dir / "fl_prediction_metrics.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()

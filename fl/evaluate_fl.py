import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from fl.model import create_model, predict


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(y_true, y_pred):
    """
    Input:
        y_true: (N, H, F_in)
        y_pred: (N, H, F_in)

    Output:
        mae, rmse, mape, r2
        Each has shape: (F_in,)
    """

    # Reshape:
    # (N, H, F_in) -> (N*H, F_in)
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    # MAE
    mae = np.mean(
        np.abs(y_true_flat - y_pred_flat),
        axis=0
    )

    # RMSE
    rmse = np.sqrt(
        np.mean(
            (y_true_flat - y_pred_flat) ** 2,
            axis=0
        )
    )

    # MAPE
    mape = np.mean(
        np.abs(y_true_flat - y_pred_flat)
        / (np.abs(y_true_flat) + 1e-6),
        axis=0
    ) * 100

    # R2
    r2 = []

    for i in range(y_true_flat.shape[1]):

        ss_res = np.sum(
            (y_true_flat[:, i] - y_pred_flat[:, i]) ** 2
        )

        ss_tot = np.sum(
            (y_true_flat[:, i] - np.mean(y_true_flat[:, i])) ** 2
        )

        r2_val = 1 - (
            ss_res / (ss_tot + 1e-10)
        )

        r2.append(r2_val)

    r2 = np.array(r2)

    return mae, rmse, mape, r2


# ============================================================
# MAIN
# ============================================================
def main():

    print("=" * 60)
    print("FL GLOBAL MODEL - NSFNET EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    base_path = Path(__file__).resolve().parent.parent

    data_dir = base_path / "data" / "preprocessed"

    model_path = (
        base_path
        / "models"
        / "fl_global_model.pt"
    )

    results_dir = base_path / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check global model
    # --------------------------------------------------------
    if not model_path.exists():

        print(
            f"\nERROR: Global model not found:\n"
            f"{model_path}"
        )

        print(
            "\nPlease train the FL model first."
        )

        return

    print(
        f"\nLoading global model:\n"
        f"{model_path}"
    )

    global_state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Load NSFNET test dataset
    # --------------------------------------------------------
    test_x_path = (
        data_dir / "nsfnet_test_X.npy"
    )

    test_y_path = (
        data_dir / "nsfnet_test_Y.npy"
    )

    if not test_x_path.exists():
        print(
            f"\nERROR: Test X not found:\n"
            f"{test_x_path}"
        )
        return

    if not test_y_path.exists():
        print(
            f"\nERROR: Test Y not found:\n"
            f"{test_y_path}"
        )
        return

    print("\nLoading NSFNET test dataset...")

    X_test = np.load(test_x_path)
    Y_test = np.load(test_y_path)

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"Y_test shape: {Y_test.shape}"
    )

    # --------------------------------------------------------
    # Determine input dimension
    # --------------------------------------------------------
    f_in = X_test.shape[-1]

    print(
        f"\nNumber of OD pairs: {f_in}"
    )

    if f_in != 182:

        print(
            f"WARNING: Expected 182 OD pairs, "
            f"but found {f_in}."
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    model = create_model(
        f_in=f_in
    )

    # --------------------------------------------------------
    # Load global FL model
    # --------------------------------------------------------
    try:

        model.load_state_dict(
            global_state_dict
        )

    except RuntimeError as e:

        print(
            "\nERROR: Global model architecture "
            "does not match NSFNET dataset."
        )

        print("\nDetails:")
        print(e)

        return

    model.eval()

    print(
        "\nGlobal model loaded successfully."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    print(
        "\nRunning prediction..."
    )

    batch_size = 64

    predictions = []

    with torch.no_grad():

        for i in range(
            0,
            len(X_test),
            batch_size
        ):

            batch_x = torch.tensor(
                X_test[
                    i:i + batch_size
                ],
                dtype=torch.float32
            )

            batch_pred = predict(
                model,
                batch_x
            )

            predictions.append(
                batch_pred.numpy()
            )

    Y_pred = np.concatenate(
        predictions,
        axis=0
    )

    print(
        f"Y_pred shape: {Y_pred.shape}"
    )

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------
    mae, rmse, mape, r2 = compute_metrics(
        Y_test,
        Y_pred
    )

    # --------------------------------------------------------
    # Aggregated metrics
    # --------------------------------------------------------
    mean_mae = np.mean(mae)
    mean_rmse = np.mean(rmse)
    mean_mape = np.mean(mape)
    mean_r2 = np.mean(r2)

    print("\n" + "=" * 60)
    print("NSFNET GLOBAL MODEL RESULTS")
    print("=" * 60)

    print(
        f"MAE :  {mean_mae:.6f}"
    )

    print(
        f"RMSE:  {mean_rmse:.6f}"
    )

    print(
        f"MAPE:  {mean_mape:.4f}%"
    )

    print(
        f"R2  :  {mean_r2:.6f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Per-OD results
    # --------------------------------------------------------
    results_list = []

    for i in range(f_in):

        source = (
            i // 13
        ) + 1

        destination_index = (
            i % 13
        )

        # Because self-pairs are removed:
        if destination_index >= source:
            destination = (
                destination_index + 2
            )
        else:
            destination = (
                destination_index + 1
            )

        results_list.append(
            {
                "OD_Idx": i + 1,
                "Source": source,
                "Destination": destination,
                "MAE": mae[i],
                "RMSE": rmse[i],
                "MAPE": mape[i],
                "R2": r2[i]
            }
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    df = pd.DataFrame(
        results_list
    )

    out_csv = (
        results_dir
        / "fl_prediction_metrics.csv"
    )

    df.to_csv(
        out_csv,
        index=False
    )

    print(
        f"\nPer-OD results saved to:\n"
        f"{out_csv}"
    )

    # --------------------------------------------------------
    # Save aggregated result
    # --------------------------------------------------------
    summary_df = pd.DataFrame(
        [
            {
                "Dataset": "NSFNET",
                "Num_OD": f_in,
                "MAE": mean_mae,
                "RMSE": mean_rmse,
                "MAPE": mean_mape,
                "R2": mean_r2
            }
        ]
    )

    summary_csv = (
        results_dir
        / "fl_prediction_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    print(
        f"Summary saved to:\n"
        f"{summary_csv}"
    )

    print("\nEvaluation completed.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
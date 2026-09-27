import pandas as pd

# ===== TAVI REĀLIE FOLDERI =====

fold_paths = {
    1: "runs/detect/train_fold12",
    2: "runs/detect/train_fold2",
    3: "runs/detect/train_fold3",
    4: "runs/detect/train_fold4",
    5: "runs/detect/train_fold5"
}

results = []

for fold, path in fold_paths.items():

    csv_path = f"{path}/results.csv"

    df = pd.read_csv(csv_path)

    last_row = df.iloc[-1]

    results.append({
        "fold": fold,
        "precision": last_row["metrics/precision(B)"],
        "recall": last_row["metrics/recall(B)"],
        "mAP50": last_row["metrics/mAP50(B)"],
        "mAP50-95": last_row["metrics/mAP50-95(B)"]
    })

results_df = pd.DataFrame(results)

print("\nPer-fold results:")
print(results_df)

print("\nMean values:")
print(results_df.mean(numeric_only=True))

print("\nStd deviation:")
print(results_df.std(numeric_only=True))
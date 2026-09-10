"""
Script de evaluación y consolidación de métricas para el benchmark.
Lee las salidas en results/raw/<model_name>.json y las compara contra ground_truth/labels.json.
Genera métricas consolidadas en results/metrics.csv.
"""

import json
import os
import pandas as pd

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcula la distancia de edición simple entre dos cadenas."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def evaluate_model_results(model_name: str, raw_results_file: str, ground_truth_file: str = "ground_truth/labels.json", return_df: bool = False):
    if not os.path.exists(raw_results_file):
        raise FileNotFoundError(f"No se encontró el archivo de resultados: {raw_results_file}")
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"No se encontró el ground truth: {ground_truth_file}")

    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    with open(raw_results_file, "r") as f:
        predictions = json.load(f)

    records = []
    for filename, pred_entry in predictions.items():
        if filename not in ground_truth:
            continue
        gt = ground_truth[filename]
        true_val = gt["label"]
        num_digits = gt["num_digits"]
        pred_val = pred_entry.get("predicted_value")
        latency = pred_entry.get("latency_seconds")
        raw_text = pred_entry.get("raw_response", "")

        is_exact_match = (pred_val == true_val) if pred_val is not None else False

        # Distancia de edición a nivel de caracteres numéricos
        true_str = str(true_val)
        pred_str = str(pred_val) if pred_val is not None else ""
        edit_dist = levenshtein_distance(true_str, pred_str)
        cer = edit_dist / max(len(true_str), 1)

        records.append({
            "image": filename,
            "true_label": true_val,
            "pred_label": pred_val,
            "num_digits": num_digits,
            "exact_match": is_exact_match,
            "edit_distance": edit_dist,
            "cer": cer,
            "latency": latency,
            "parse_error": pred_val is None
        })

    df = pd.DataFrame(records)

    # Métricas agregadas
    total_samples = len(df)
    exact_match_acc = df["exact_match"].mean()
    mean_cer = df["cer"].mean()
    mean_latency = df["latency"].mean()
    median_latency = df["latency"].median()
    parse_error_rate = df["parse_error"].mean()

    # Desglose por cantidad de dígitos
    acc_1_digit = df[df["num_digits"] == 1]["exact_match"].mean()
    acc_2_digit = df[df["num_digits"] == 2]["exact_match"].mean()
    acc_3_digit = df[df["num_digits"] == 3]["exact_match"].mean()
    acc_zero = df[df["true_label"] == 0]["exact_match"].mean()

    summary = {
        "model": model_name,
        "samples": total_samples,
        "exact_match_accuracy": round(exact_match_acc, 4),
        "mean_cer": round(mean_cer, 4),
        "acc_zeros": round(acc_zero, 4),
        "acc_1_digit": round(acc_1_digit, 4),
        "acc_2_digits": round(acc_2_digit, 4),
        "acc_3_digits": round(acc_3_digit, 4),
        "mean_latency_s": round(mean_latency, 3) if mean_latency is not None else None,
        "median_latency_s": round(median_latency, 3) if median_latency is not None else None,
        "parse_error_rate": round(parse_error_rate, 4)
    }

    # Guardar / actualizar metrics.csv
    metrics_file = "results/metrics.csv"
    os.makedirs("results", exist_ok=True)
    summary_df = pd.DataFrame([summary])

    if os.path.exists(metrics_file):
        existing = pd.read_csv(metrics_file)
        # Reemplazar si el modelo ya existía, o agregar si es nuevo
        existing = existing[existing["model"] != model_name]
        updated = pd.concat([existing, summary_df], ignore_index=True)
        updated.to_csv(metrics_file, index=False)
    else:
        summary_df.to_csv(metrics_file, index=False)

    print(f"=== Métricas para {model_name} ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Consolidado guardado en {metrics_file}")

    # Guardar reporte comparativo enriquecido
    os.makedirs("results/evaluated", exist_ok=True)
    comparison_csv = f"results/evaluated/{model_name}_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    # Generar visualización gráfica de errores
    error_rows = df[~df["exact_match"]]
    error_plot_path = None
    if len(error_rows) > 0:
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            n_err = len(error_rows)
            cols = min(4, n_err)
            rows = (n_err + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.2))
            axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for idx, (_, row) in enumerate(error_rows.iterrows()):
                img_path = os.path.join("images/test", row["image"])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes_list[idx].imshow(img, cmap="gray")
                axes_list[idx].set_title(
                    f"Real: {row['true_label']} | Pred: {row['pred_label']}\n({row['image']})",
                    color="darkred",
                    fontsize=9,
                    fontweight="bold"
                )
                axes_list[idx].axis("off")

            for extra in range(n_err, len(axes_list)):
                axes_list[extra].axis("off")

            plt.tight_layout()
            error_plot_path = f"results/evaluated/{model_name}_errors.png"
            plt.savefig(error_plot_path, dpi=130, bbox_inches="tight")
            plt.close()
        except Exception as err:
            print(f"No se pudo generar la grilla visual de errores: {err}")

    # Registro en MLflow (Experiment Tracking)
    try:
        import mlflow
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        mlflow.set_experiment("telegramas-vlm-benchmark")
        with mlflow.start_run(run_name=model_name):
            # 1. Parámetros del benchmark
            mlflow.log_params({
                "model_name": model_name,
                "total_samples": total_samples,
                "raw_results_file": raw_results_file,
                "ground_truth_file": ground_truth_file
            })

            # 2. Métricas numéricas
            metrics_to_log = {
                "exact_match_accuracy": float(summary["exact_match_accuracy"]),
                "mean_cer": float(summary["mean_cer"]),
                "parse_error_rate": float(summary["parse_error_rate"])
            }
            if summary["mean_latency_s"] is not None and not pd.isna(summary["mean_latency_s"]):
                metrics_to_log["mean_latency_s"] = float(summary["mean_latency_s"])
            if not pd.isna(summary["acc_zeros"]):
                metrics_to_log["acc_zeros"] = float(summary["acc_zeros"])
            if not pd.isna(summary["acc_1_digit"]):
                metrics_to_log["acc_1_digit"] = float(summary["acc_1_digit"])
            if not pd.isna(summary["acc_2_digits"]):
                metrics_to_log["acc_2_digits"] = float(summary["acc_2_digits"])
            if not pd.isna(summary["acc_3_digits"]):
                metrics_to_log["acc_3_digits"] = float(summary["acc_3_digits"])

            mlflow.log_metrics(metrics_to_log)

            # 3. Artefactos: JSON crudo, CSV comparativo y Grilla visual de errores
            if os.path.exists(raw_results_file):
                mlflow.log_artifact(raw_results_file, artifact_path="raw_predictions")
            if os.path.exists(comparison_csv):
                mlflow.log_artifact(comparison_csv, artifact_path="evaluation_reports")
            if error_plot_path and os.path.exists(error_plot_path):
                mlflow.log_artifact(error_plot_path, artifact_path="evaluation_reports")

        print("Métricas, reporte comparativo y gráfico de errores registrados exitosamente en MLflow.")
    except Exception as e:
        print(f"MLflow tracking omitido o con advertencia: {e}")
    if return_df:
        return summary, df
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        evaluate_model_results(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python src/evaluate.py <nombre_modelo> <ruta_a_raw_results.json>")

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

def evaluate_model_results(model_name: str, raw_results_file: str, ground_truth_file: str = "ground_truth/labels.json"):
    if not os.path.exists(raw_results_file):
        raise FileNotFoundError(f"No se encontró el archivo de resultados: {raw_results_file}")
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"No se encontró el ground truth: {ground_truth_file}")

    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    with open(raw_results_file, "r") as f:
        predictions = json.load(f)

    records = []
    for filename, gt in ground_truth.items():
        true_val = gt["label"]
        num_digits = gt["num_digits"]
        pred_entry = predictions.get(filename, {})
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
    return summary

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        evaluate_model_results(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python src/evaluate.py <nombre_modelo> <ruta_a_raw_results.json>")

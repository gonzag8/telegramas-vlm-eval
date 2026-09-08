"""
Ejecutor del benchmark para Qwen-VL vía OpenRouter.
Utiliza el SDK de OpenAI configurado con la base URL de OpenRouter.
Guarda las predicciones crudas en results/raw/qwen_vl.json.
"""

import os
import json
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from prompts import build_messages

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen/qwen-2.5-vl-72b-instruct:free")

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_model_response(raw_content: str):
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        val = data.get("value")
        if val is not None:
            return int(val)
        return None
    except Exception:
        import re
        match = re.search(r'\b\d+\b', text)
        if match:
            return int(match.group(0))
        return None

def run_benchmark(images_dir: str = "images/test", output_file: str = "results/raw/qwen_vl.json"):
    if not OPENROUTER_API_KEY:
        raise ValueError("No se encontró OPENROUTER_API_KEY en el entorno ni en el archivo .env.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    image_paths = sorted(list(Path(images_dir).glob("*.png")))
    
    print(f"Iniciando benchmark Qwen-VL ({MODEL_NAME}) sobre {len(image_paths)} imágenes...")
    
    results = {}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        filename = img_path.name
        img_b64 = encode_image_base64(str(img_path))
        messages = build_messages(img_b64)

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            elapsed = time.time() - start_time
            raw_content = response.choices[0].message.content
            pred_value = parse_model_response(raw_content)

            results[filename] = {
                "predicted_value": pred_value,
                "raw_response": raw_content,
                "latency_seconds": elapsed,
                "model": MODEL_NAME,
                "status": "success"
            }
            print(f"[{i:02d}/{len(image_paths)}] {filename} -> {pred_value} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{i:02d}/{len(image_paths)}] ERROR en {filename}: {e}")
            results[filename] = {
                "predicted_value": None,
                "raw_response": str(e),
                "latency_seconds": elapsed,
                "model": MODEL_NAME,
                "status": "error"
            }
            time.sleep(2)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completado. Resultados guardados en {output_file}")

if __name__ == "__main__":
    run_benchmark()

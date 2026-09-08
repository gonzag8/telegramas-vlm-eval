"""
Ejecutor del benchmark para Llama Vision vía Groq API.
Procesa las imágenes en images/test/ usando el prompt unificado de src/prompts.py.
Guarda las predicciones crudas en results/raw/llama_vision.json.
"""

import os
import json
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from prompts import build_messages

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_model_response(raw_content: str):
    """
    Intenta extraer el valor numérico del JSON retornado.
    Soporta si el modelo añade backticks markdown ```json ... ```.
    """
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
        # Fallback: intentar encontrar el primer número entero en el texto
        import re
        match = re.search(r'\b\d+\b', text)
        if match:
            return int(match.group(0))
        return None

def run_benchmark(images_dir: str = "images/test", output_file: str = "results/raw/llama_vision.json"):
    if not GROQ_API_KEY:
        raise ValueError("No se encontró GROQ_API_KEY en el entorno ni en el archivo .env.")

    client = Groq(api_key=GROQ_API_KEY)
    image_paths = sorted(list(Path(images_dir).glob("*.png")))
    
    print(f"Iniciando benchmark Groq ({MODEL_NAME}) sobre {len(image_paths)} imágenes...")
    
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
                temperature=0.0, # Determinismo para evaluación justa
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
            # Evitar rate limits si ocurre error
            time.sleep(1)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completado. Resultados guardados en {output_file}")

if __name__ == "__main__":
    run_benchmark()

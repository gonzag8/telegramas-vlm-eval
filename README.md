# VLM Benchmark — Dígitos escritos a mano

Ambiente de pruebas para evaluar Vision Language Models (Chandra-OCR, Qwen-VL, Llama Vision) aplicados a la lectura de Dígitos escritos a mano.

**Estado actual**: en configuración inicial. Todavía no hay resultados — este README documenta el diseño del ambiente de pruebas antes de correr el primer benchmark.

## Objetivo

Comparar qué tan bien distintos VLMs extraen datos estructurados a partir de imágenes de planillas, usando el mismo set de imágenes y el mismo prompt para cada modelo, de forma que la comparación sea justa entre ellos.

## Modelos a evaluar

| Modelo | Vía de acceso | Por qué se incluye |
|---|---|---|
| Chandra-OCR 2 | Hosted API / playground | Especializado en documentos estructurados y tablas |
| Qwen-VL | OpenRouter | Modelo generalista con buen grounding multimodal |
| Llama Vision | Groq API | Punto de comparación como modelo generalista alternativo |

## Estructura del repositorio

```
vlm-ocr-telegramas/
├── README.md
├── requirements.txt
├── .env.example
├── images/
│   └── test/              # imágenes de prueba (no sensibles)
├── ground_truth/
│   └── labels.json        # valores correctos por imagen, para medir accuracy
├── notebooks/
│   ├── 01_chandra_ocr.ipynb
│   ├── 02_qwen_vl.ipynb
│   └── 03_llama_vision.ipynb
├── src/
│   ├── prompts.py          # prompt único, compartido entre los tres modelos
│   └── evaluate.py         # compara salidas contra ground truth y calcula accuracy
└── results/
    ├── raw/                # salida cruda de cada modelo, por imagen
    └── metrics.csv         # accuracy y tiempo de respuesta, consolidado
```

## Metodología

1. Se arma un set fijo de imágenes de prueba con su ground truth (valores correctos conocidos).
2. Se define un único prompt, adaptado a la sintaxis de cada API, pidiendo siempre el mismo formato de salida en JSON.
3. Cada modelo procesa el mismo set de imágenes con el mismo prompt.
4. Se compara cada salida contra el ground truth, campo por campo, y se calcula accuracy y tiempo de respuesta por modelo.
5. Los resultados se documentan en este README a medida que están disponibles.

## Cómo correrlo

```bash
git clone <url-del-repo>
cd telegramas-vlm-eval
pip install -r requirements.txt
cp .env.example .env   # completar con las API keys correspondientes
```

Las API keys necesarias (Groq, OpenRouter, y la de Datalab si se usa la hosted API en vez del playground) se completan en `.env`, que no se sube al repositorio.

Cada notebook en `notebooks/` es independiente y corre el benchmark para un modelo. `src/evaluate.py` consolida los resultados de `results/raw/` en `results/metrics.csv`.

## Resultados

| Modelo | Modo | Muestras | Exact Match | CER (Mean) | Ceros (0) | 1 Dígito | 2 Dígitos | 3 Dígitos | Latencia Media |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-VL-72B** | Zero-Shot (Baseline) | 50 | **76.0%** | 0.1933 | 90.0% | 80.0% | 66.7% | 80.0% | 1.03 s |
| **Qwen2.5-VL-72B** | Few-Shot (Domain-Rules) | 50 | **80.0%** (87% efect.) | 0.1700 | 90.0% | 88.0% | 66.7% | 80.0% | 5.58 s |

### Experiment Tracking con MLflow
El proyecto cuenta con seguimiento automático de experimentos mediante **MLflow**.
Para explorar la interfaz interactiva, gráficos comparativos y auditoría visual de errores:
```bash
mlflow ui
```
Navegar a `http://localhost:5000` para ver métricas, matrices comparativas y las imágenes recortadas con fallos de predicción.

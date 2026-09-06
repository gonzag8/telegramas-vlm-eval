# VLM Benchmark — Telegramas Electorales

Ambiente de pruebas para evaluar VLMs (Chandra-OCR, Qwen-VL, Llama Vision) aplicados a la lectura de telegramas electorales de la Provincia de Santa Fe.

## Objetivo

Comparar qué tan bien distintos Vision Language Models extraen datos estructurados (votos por partido, número de mesa, etc.) de imágenes de planillas, usando el mismo set de imágenes y el mismo prompt para cada uno.

---

## Paso a paso para dejar el repo listo

### 1. Crear el repositorio

- Nombre sugerido: `vlm-benchmark-telegramas` (o el que prefiera el profe)
- Privado, dado que eventualmente va a tener imágenes de datos electorales
- Invitar como colaboradores al profe y a quien más aporte del equipo

### 2. Estructura de carpetas

```
vlm-benchmark-telegramas/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── images/
│   └── test/              # imágenes de prueba, NO sensibles al inicio
├── ground_truth/
│   └── labels.json        # valores correctos por imagen
├── notebooks/
│   ├── 01_chandra_ocr.ipynb
│   ├── 02_qwen_vl.ipynb
│   └── 03_llama_vision.ipynb
├── src/
│   ├── prompts.py          # prompt único compartido entre modelos
│   └── evaluate.py         # script de comparación contra ground truth
└── results/
    ├── raw/                # salida cruda de cada modelo, por imagen
    └── metrics.csv         # accuracy, tiempo, etc. consolidado
```

### 3. Configurar el entorno

`requirements.txt` mínimo para arrancar (vas a sumar más según lo que necesite cada notebook):

```
python-dotenv
requests
groq
openai          # OpenRouter usa el mismo cliente que OpenAI
pillow
pandas
jupyter
```

`.env.example` (nunca subir el `.env` real, solo este template):

```
GROQ_API_KEY=
OPENROUTER_API_KEY=
DATALAB_API_KEY=
```

`.gitignore` — como mínimo:

```
.env
__pycache__/
*.ipynb_checkpoints
images/real/           # cuando haya imágenes reales, nunca al repo
```

### 4. Conseguir las API keys

- **Groq** (Llama Vision): cuenta gratis en console.groq.com, tiene free tier generoso
- **OpenRouter** (Qwen-VL): cuenta gratis en openrouter.ai, créditos gratis para empezar
- **Datalab** (Chandra-OCR): revisar si el playground gratuito alcanza para el volumen de pruebas, o si hace falta la hosted API con key

### 5. Preparar el ground truth

- Elegir 5-10 imágenes de telegramas de prueba (no sensibles, o mock generadas)
- Armar `ground_truth/labels.json` a mano con el valor correcto de cada campo por imagen, ejemplo:

```json
{
  "telegrama_01.png": {
    "mesa": "0123",
    "partido_A": 145,
    "partido_B": 98,
    "partido_C": 12
  }
}
```

Sin esto no se puede medir accuracy real, solo comparar visualmente.

### 6. Definir el prompt único (`src/prompts.py`)

Mismo prompt para los tres modelos (adaptado a la sintaxis de cada API), pidiendo siempre el mismo formato de salida en JSON, para que la comparación sea justa.

### 7. Notebook por modelo

Cada notebook (`01_chandra_ocr.ipynb`, `02_qwen_vl.ipynb`, `03_llama_vision.ipynb`) debería:

1. Cargar las imágenes de `images/test/`
2. Llamar al modelo correspondiente con el prompt de `src/prompts.py`
3. Guardar la salida cruda en `results/raw/<modelo>/<imagen>.json`

### 8. Script de evaluación (`src/evaluate.py`)

Compara cada salida en `results/raw/` contra `ground_truth/labels.json`, calcula accuracy por campo y por modelo, y vuelca todo en `results/metrics.csv`.

### 9. README con hallazgos (actualizar esta sección a medida que hay resultados)

Tabla final con accuracy, tiempo de respuesta promedio, y ejemplos concretos de dónde falla cada modelo — esto es lo que más le interesa ver al profe, más que el código en sí.

---

## Próximos pasos (una vez armado)

- [ ] Correr Chandra-OCR sobre el set de prueba
- [ ] Correr Qwen-VL sobre el mismo set
- [ ] Correr Llama Vision sobre el mismo set
- [ ] Consolidar métricas y comparar
- [ ] Confirmar con el profe la política de datos antes de correr sobre telegramas reales (privacidad de datos electorales — evaluar si conviene inferencia local/on-premise en vez de APIs de terceros)

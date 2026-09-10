"""
Prompts compartidos para el benchmark de VLMs sobre números manuscritos.
Garantiza que todos los modelos reciban exactamente la misma consigna y contexto.
"""

SYSTEM_PROMPT = """Eres un sistema de OCR especializado en digitalización de telegramas y actas electorales.
Tu tarea es reconocer con máxima precisión el número manuscrito que aparece en el recorte de la imagen."""

USER_PROMPT = """Examina la imagen y extrae el número manuscrito de la casilla electoral.
Reglas del dominio electoral:
1. Formato numérico: El contenido es un entero no negativo (0 a 999).
2. Ceros a la izquierda (padding): En planillas electorales es habitual rellenar las casillas con ceros a la izquierda (ej. "009", "03", "000"). Interpreta esos trazos circulares iniciales como ceros y extrae el valor entero final (ej: "009" -> 9; "03" -> 3; "000" -> 0).
3. Dígito 7: El 7 manuscrito suele escribirse con una barra horizontal cruzada (estilo latino). Ten cuidado de no confundirlo con un 4.
4. Dígito 8 vs 3: Presta especial atención al cierre del trazo izquierdo del 8 para distinguirlo de un 3.
5. Marcas protectoras: Ignora guiones, barras o líneas de seguridad antes o después del número (ej. "- 13 -" -> 13). No devuelvas números negativos.
6. Formato de salida: Responde ÚNICAMENTE con un objeto JSON válido con la clave "value".
   - Ejemplo: {"value": 42}
   - Si la celda está vacía o es ilegible: {"value": null}
No incluyas explicaciones, texto adicional ni bloques markdown fuera del JSON."""

import base64
import os
from functools import lru_cache

@lru_cache(maxsize=10)
def _get_few_shot_b64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

def build_messages(image_base64: str, mime_type: str = "image/png", few_shot: bool = False):
    """
    Construye la estructura de mensajes estándar compatible con OpenAI/OpenRouter.
    Soporta modo Zero-Shot y Few-Shot con ejemplos visuales de dominio electoral.
    """
    if not few_shot:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

    # Modo Few-Shot con ejemplos visuales en contexto
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Ejemplo 1: El 7 manuscrito con barra cruzada
    ex_seven = _get_few_shot_b64("images/few_shot/example_seven.png")
    if ex_seven:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Ejemplo de referencia: casilla con número manuscrito. Observa que el 7 suele llevar barra horizontal."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex_seven}"}}
            ]
        })
        messages.append({"role": "assistant", "content": '{"value": 7}'})

    # Ejemplo 2: Cifra de un dígito con ceros / trazo rápido
    ex_nine = _get_few_shot_b64("images/few_shot/example_nine.png")
    if ex_nine:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Ejemplo de referencia: casilla con dígito manuscrito. Ignora ceros de relleno o marcas previas."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ex_nine}"}}
            ]
        })
        messages.append({"role": "assistant", "content": '{"value": 9}'})

    # Consulta objetivo actual
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": USER_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        ]
    })
    return messages

"""
Prompts compartidos para el benchmark de VLMs sobre números manuscritos.
Garantiza que todos los modelos reciban exactamente la misma consigna y contexto.
"""

SYSTEM_PROMPT = """Eres un sistema de OCR especializado en digitalización de telegramas y actas electorales.
Tu tarea es reconocer con máxima precisión el número manuscrito que aparece en el recorte de la imagen."""

USER_PROMPT = """Examina la imagen y extrae el número manuscrito.
Reglas:
1. El contenido es un número entero (puede ser 0, de un dígito, dos dígitos o tres dígitos).
2. Responde ÚNICAMENTE con un objeto JSON válido con la clave "value".
3. Ejemplo de respuesta: {"value": 42}
4. Si la celda está vacía o es ilegible, responde: {"value": null}
No incluyas explicaciones, texto adicional ni bloques markdown fuera del JSON."""

def build_messages(image_base64: str, mime_type: str = "image/png"):
    """
    Construye la estructura de mensajes estándar compatible con Groq y OpenRouter (OpenAI-compatible).
    """
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

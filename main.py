from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import os
import json
import base64
import openai

# Создаём клиента OpenAI
client = openai.OpenAI()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    full_text: str
    regions: list

@app.post("/upload")
async def analyze_image(
    file: UploadFile = File(...),
    user_level: str = Form(...),
    detailed: str = Form(...),
):
    try:
        print("==> Обработка запроса начата")
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((768, 768))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        detailed_flag = detailed.lower() == "true"
        prompt = build_prompt(user_level, detailed_flag)

        print("==> Отправка запроса в OpenAI...")

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1200
        )

        content = response.choices[0].message.content
        print("==> Ответ получен")
        print("==> Контент:", content[:200], "...")

        extracted_json = extract_json(content)

        return {
            "full_text": content.strip(),
            "regions": extracted_json.get("regions", [])
        }

    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


def extract_json(text):
    try:
        json_start = text.index("{")
        json_str = text[json_start:]
        return json.loads(json_str)
    except Exception as e:
        print("==> Ошибка парсинга JSON:", e)
        return {"regions": []}


def build_prompt(user_level, detailed):
    intro = f"""Проанализируй загруженную фотографию (в base64 ниже) как профессиональный фотокритик. Представь, что ты общаешься с фотографом уровня: {user_level}.

Твоя задача — дать отзыв в вежливом, но честном стиле. Укажи:

1. Композиция
2. Свет и цвет
3. История или эмоция
4. Технические параметры
5. Общая оценка (по 10-балльной шкале)
6. Совет
"""

    region_hint = """
Если включён расширенный анализ, добавь JSON-блок:
{
  "regions": [
    {"x": 0.3, "y": 0.2, "width": 0.1, "height": 0.2, "comment": "Комментарий"}
  ]
}
где координаты — относительные (в долях от 0 до 1). Не добавляй ключ summary.
"""

    return intro + (region_hint if detailed else "")
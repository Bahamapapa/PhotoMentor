from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import openai
import base64
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

class FeedbackRequest(BaseModel):
    image_base64: str
    user_level: str
    detailed: bool = False

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), user_level: str = Form(...), detailed: bool = Form(False)):
    print("==> Обработка запроса начата")

    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")
    content = base64_image

    try:
        print("==> Отправка запроса в OpenAI...")

        prompt = f"""
Вы — эксперт по визуальному анализу фотографий. Проведи разбор изображения по следующим пунктам:

1. **Композиция**
2. **Свет и цвет**
3. **История или эмоция**
4. **Технические параметры**
5. **Оценка** — включи короткое текстовое обоснование + числовую оценку по шкале от 1 до 10
6. **Совет** — краткая рекомендация по улучшению фото

Пиши кратко, но информативно, на русском языке.

Уровень зрителя: {user_level}.

{f"Также выдели максимум 3 участков изображения, нуждающихся в улучшении, и верни их координаты в процентах от размеров изображения (от 0 до 1) в JSON-формате под этим текстом. Каждый участок должен:
- точно ограничивать нужный объект, не выходить за его контуры,
- не включать лишний фон,
- быть компактным (только по размеру проблемной области),
- не объединять несколько объектов в один блок.
Пример: \"x\": 0.1, \"y\": 0.2, \"width\": 0.3, \"height\": 0.2, \"comment\": \"Комментарий\"." if detailed else ""}
"""

        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{content}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1400,
        )

        raw_text = response.choices[0].message.content
        print("==> Ответ получен")
        print("==> Контент:", raw_text[:300])

        feedback_text = raw_text.strip()
        extracted_json = {}

        if detailed:
            json_start = feedback_text.find("{"),
            json_end = feedback_text.rfind("}") + 1
            if json_start and json_end:
                try:
                    extracted_json = json.loads(feedback_text[json_start[0]:json_end])
                except Exception as e:
                    print("==> Ошибка при разборе JSON:", e)

        feedback = {
            "full_text": feedback_text,
            **({"regions": extracted_json.get("regions", [])} if detailed else {})
        }

        print("==> Извлечённый JSON:", extracted_json if extracted_json else "(нет)")
        return {"feedback": feedback}

    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", str(e))
        return {"error": str(e)}
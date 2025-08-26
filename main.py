import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackResponse(BaseModel):
    feedback: Optional[str] = None
    regions: Optional[list] = None

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_level: str = Form(...), detailed: bool = Form(...)):
    print("==> Обработка запроса начата")

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = f"""
Ты — визуальный критик фотографии. Проанализируй прикреплённое фото.

Если пользователь включил расширенный анализ, сначала выдели до 5 участков на изображении, в которых можно что-то улучшить. Для каждого участка укажи:
- координаты `x`, `y`, `width`, `height` (от 0 до 1, два знака после запятой)
- краткий, конкретный комментарий, что стоит улучшить

Затем выполни текстовый анализ фото по следующим пунктам:
1. Композиция
2. Свет и цвет
3. История или эмоция
4. Технические параметры

Добавь **оценку по десятибалльной шкале** в виде: **Оценка: 8/10**

Заверши кратким **советом**, что можно улучшить.

Формат ответа: JSON. Пример:
```json
{{
  "full_text": "<полный анализ, включая Оценка: 8/10 и совет>",
  "regions": [
    {{
      "x": 0.25,
      "y": 0.3,
      "width": 0.15,
      "height": 0.2,
      "comment": "Комментарий к проблемной зоне"
    }}
  ]
}}
```

Если расширенный анализ не включён, верни только `"full_text"`.

Уровень пользователя: {user_level}
Расширенный анализ: {"да" if detailed else "нет"}

Изображение (base64): data:image/jpeg;base64,{img_str}
"""

    print("==> Отправка запроса в OpenAI...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )
    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", e)
        return {"error": str(e)}

    content = response.choices[0].message.content
    print("==> Ответ получен")
    print("==> Контент:", content)

    try:
        json_start = content.index("{")
        json_data = content[json_start:]
        feedback = eval(json_data)
    except Exception as e:
        feedback = {"full_text": content}
        print("==> Не удалось распарсить JSON, используем как текст")

    return {"feedback": feedback}
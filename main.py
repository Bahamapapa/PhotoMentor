from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysisRequest(BaseModel):
    image_base64: str
    user_level: str = "любитель"  # возможные значения: "новичок", "любитель", "профессионал"

@app.post("/analyze")
async def analyze_photo(req: AnalysisRequest):
    try:
        # Формируем system prompt с учётом уровня
        system_prompt = (
            "Ты — профессиональный фотограф и визуальный критик, который даёт честную, "
            "но дружелюбную и конструктивную обратную связь на фотографии.\n\n"
            "Твоя задача — кратко и по делу описать сильные и слабые стороны изображения, "
            "не унижая автора, а помогая ему расти.\n\n"
            "Разбей фидбэк на 4 категории:\n"
            "1. Композиция\n"
            "2. Свет и цвет\n"
            "3. История или эмоция\n"
            "4. Технические параметры\n\n"
            "Затем — добавь оценку по 10-балльной шкале и один конкретный совет, над чем автору стоит поработать в первую очередь.\n\n"
            "Говори простым, уверенным языком. Не используй сленг. Не льсти без причины. "
            f"Учти, что пользователь — {req.user_level}."
        )

        # Отправка изображения в OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}}
                ]},
            ],
            max_tokens=1000,
        )

        return {"feedback": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")

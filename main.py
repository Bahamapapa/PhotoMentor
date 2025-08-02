import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
from openai import OpenAI

load_dotenv()

app = FastAPI(
    title="PhotoMentor API",
    description="Анализ фотографии через OpenAI Vision",
    version="1.0.0"
)

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"message": "PhotoMentor backend работает"}

@app.post("/analyze")
async def analyze_photo(
    file: UploadFile = File(...),
    user_level: str = Form("любитель")
):
    """
    Анализ фотографии с помощью GPT-4o Vision.
    Принимает изображение и уровень пользователя (новичок, любитель, профи).
    """
    try:
        # Читаем и кодируем в base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Системный промпт
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
            f"Учти, что пользователь — {user_level}."
        )

        # Отправка в OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]}
            ],
            max_tokens=1000
        )

        feedback = response.choices[0].message.content
        return {"feedback": feedback}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")
import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(
    title="PhotoMentor API",
    description="Анализ фотографии через OpenAI Vision",
    version="1.0.0"
)

# CORS
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

class AnalysisRequest(BaseModel):
    image_base64: str
    user_level: str = "любитель"

def generate_feedback(base64_image: str, user_level: str) -> str:
    # НОВЫЙ СИСТЕМНЫЙ ПРОМПТ
    system_prompt = (
        "Ты — профессиональный фотограф и визуальный критик. Твоя задача — дать честную, "
        "конструктивную, но не уничижительную обратную связь на фотографию.\n"
        "Пиши кратко и по делу, помогая автору расти, а не хваля без причины.\n"
        "Используй форматирование жирным шрифтом Markdown (**текст**).\n"
        "Структурируй ответ строго по следующему шаблону, используя переносы строк:\n"
        "\n"
        "**1. Композиция:** (Текст об анализе композиции)\n"
        "\n"
        "**2. Свет и цвет:** (Текст об анализе света и цвета)\n"
        "\n"
        "**3. История или эмоция:** (Текст об анализе истории или эмоций)\n"
        "\n"
        "**4. Технические параметры:** (Текст об анализе технических параметров)\n"
        "\n"
        "**Оценка:** (Оценка от 1 до 10)\n"
        "\n"
        "**Совет:** (Один конкретный совет, с чего автору лучше начать улучшения)\n"
        "\n"
        f"Учти, что пользователь — {user_level}."
    )

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

    return response.choices[0].message.content

@app.post("/analyze")
async def analyze_base64(req: AnalysisRequest):
    try:
        feedback = generate_feedback(req.image_base64, req.user_level)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_level: str = Form("любитель")
):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        feedback = generate_feedback(base64_image, user_level)
        return {"feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке: {str(e)}")
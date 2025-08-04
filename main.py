import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
from openai import OpenAI
import re

load_dotenv()

app = FastAPI(
    title="PhotoMentor API",
    description="Анализ фотографии через OpenAI Vision",
    version="1.1.0"
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

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_level: str = Form("любитель"),
    detailed: bool = Form(False)
):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        system_prompt = (
            "Ты — профессиональный фотограф и визуальный критик."
            " Дай честную, конструктивную обратную связь.
            "Разбей её на 4 категории: Композиция, Свет и цвет, История, Технические параметры.
            "Поставь оценку по 10-балльной шкале и один главный совет."
        )

        if detailed:
            system_prompt += (
                "\n\nЕсли возможно, выдели на фото до 5 ключевых участков, которые можно улучшить,"
                " используя красные прямоугольники. Обозначь их как Участок 1, Участок 2 и т.д."
                "\n\nТекстом под изображением кратко опиши проблему для каждого участка."
                "\n\nФормат — относительные координаты (в процентах)."
                "\n\nПример описания участков:"
                "\n• Участок 1 (центр кадра): основной объект смазан, фокус не попал."
                "\n• Участок 2 (левый верхний угол): неинформативная зона, отвлекает внимание."
                "\n\nВерни результат в тексте, где участки описаны в следующем формате:"
                "\n[Участок 1] x=10%, y=20%, w=30%, h=15% — короткий комментарий."
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1500
        )

        feedback = response.choices[0].message.content.strip()
        regions = []

        if detailed:
            region_pattern = re.compile(r"\[Участок (\d+)\]\s+x=(\d+)%\s*,\s*y=(\d+)%\s*,\s*w=(\d+)%\s*,\s*h=(\d+)%\s*—\s*(.+)")
            for match in region_pattern.finditer(feedback):
                regions.append({
                    "label": f"Участок {match.group(1)}",
                    "x": int(match.group(2)),
                    "y": int(match.group(3)),
                    "width": int(match.group(4)),
                    "height": int(match.group(5)),
                    "comment": match.group(6).strip()
                })

        return {"feedback": feedback, "regions": regions if detailed else []}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")
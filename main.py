import os
import base64
import json
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(
    title="PhotoMentor API",
    description="Анализ фотографии через OpenAI Vision",
    version="2.3.0"
)

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
        print("==> Обработка запроса начата")
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        if detailed:
            system_prompt = (
                "Ты — профессиональный фотограф и визуальный критик.\n"
                "Сначала проанализируй изображение и выдели до 5 ключевых участков, которые требуют улучшения.\n"
                "Верни JSON-объект **в самом начале** ответа в блоке ```json``` следующей структуры:\n\n"
                "{\n"
                "  \"score\": 9.0,\n"
                "  \"regions\": [\n"
                "    {\"x\": 0.1, \"y\": 0.2, \"width\": 0.3, \"height\": 0.2, \"comment\": \"описание проблемы\"},\n"
                "    ...\n"
                "  ]\n"
                "}\n\n"
                "После этого дай полноценный анализ фотографии, согласованный с замечаниями в JSON. Ответ строго по шаблону в Markdown:\n\n"
                "**1. Композиция:**\n\n"
                "**2. Свет и цвет:**\n\n"
                "**3. История или эмоция:**\n\n"
                "**4. Технические параметры:**\n\n"
                "**Оценка:**\n\n"
                "**Совет:**\n\n"
                f"Уровень пользователя: {user_level}."
            )
        else:
            system_prompt = (
                "Ты — профессиональный фотограф и визуальный критик.\n"
                "Дай краткую, конструктивную обратную связь на фотографию, используя Markdown.\n"
                "Ответ строго по шаблону:\n\n"
                "**1. Композиция:**\n\n"
                "**2. Свет и цвет:**\n\n"
                "**3. История или эмоция:**\n\n"
                "**4. Технические параметры:**\n\n"
                "**Оценка:**\n\n"
                "**Совет:**\n\n"
                f"Уровень пользователя: {user_level}."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        print("==> Отправка запроса в OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1800
        )

        print("==> Ответ получен")
        content = response.choices[0].message.content.strip()
        print("==> Контент:", content[:200] + "..." if len(content) > 200 else content)

        feedback_text = content
        regions_data = {}

        if detailed:
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                json_block = match.group(1)
                try:
                    regions_data = json.loads(json_block)
                    feedback_text = content.replace(match.group(0), '').strip()
                except json.JSONDecodeError as e:
                    print("==> Ошибка JSONDecode:", e)

        return JSONResponse(content={
            "feedback": {
                "full_text": feedback_text,
                "regions": regions_data.get("regions", []),
                "score": regions_data.get("score")
            }
        })

    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
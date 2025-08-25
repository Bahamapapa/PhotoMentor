import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(
    title="PhotoMentor API",
    description="Анализ фотографии через OpenAI Vision",
    version="2.0.0"
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
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        system_prompt = (
            "Ты — профессиональный фотограф и визуальный критик. "
            "Дай краткую, конструктивную обратную связь на фотографию, используя Markdown. "
            "Ответ структурируй строго по следующему шаблону:\n\n"
            "**1. Композиция:**\n\n"
            "**2. Свет и цвет:**\n\n"
            "**3. История или эмоция:**\n\n"
            "**4. Технические параметры:**\n\n"
            "**Оценка:**\n\n"
            "**Совет:**\n\n"
            f"Учти, что пользователь — {user_level}."
        )

        if detailed:
            system_prompt += (
                "\n\nЕсли возможно, выдели до 5 ключевых участков, которые можно улучшить. "
                "Верни JSON-объект вида:\n\n"
                "{\n"
                '  "summary": "текст общей обратной связи",\n'
                '  "regions": [\n'
                "    {\"x\": 0.4, \"y\": 0.5, \"width\": 0.2, \"height\": 0.1, \"comment\": \"проблема\"},\n"
                "    ...\n"
                "  ]\n"
                "}"
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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1800,
            response_format="json" if detailed else "text"
        )

        if detailed:
            raw = response.choices[0].message.content.strip()
            structured = eval(raw) if isinstance(raw, str) else raw
            return JSONResponse(content={"feedback": structured})
        else:
            feedback = response.choices[0].message.content.strip()
            return JSONResponse(content={"feedback": {"summary": feedback, "regions": []}})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")
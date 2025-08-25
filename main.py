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
    version="2.2.0"
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
            f"Уровень пользователя: {user_level}."
        )

        if detailed:
            system_prompt += (
                "\n\nЕсли возможно, выдели до 5 ключевых участков, которые можно улучшить. "
                "Верни JSON-объект в конце ответа в следующей структуре в блоке ```json ...```:\n\n"
                '{\n'
                '  "regions": [\n'
                '    {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.2, "comment": "описание"},\n'
                '    ...\n'
                '  ]\n'
                '}'
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

        if detailed:
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                json_block = match.group(1)
            else:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_block = content[json_start:json_end] if json_start != -1 and json_end > json_start else ''

            print("==> Извлечённый JSON:", json_block[:200])

            try:
                parsed = json.loads(json_block)
                assert isinstance(parsed, dict)
                assert "regions" in parsed
                return JSONResponse(content={
                    "feedback": {
                        "regions": parsed.get("regions", []),
                        "full_text": content
                    }
                })
            except Exception as parse_err:
                print("==> Ошибка разбора JSON:", parse_err)
                return JSONResponse(content={
                    "feedback": {
                        "regions": [],
                        "full_text": content
                    }
                })

        return JSONResponse(content={"feedback": {"regions": [], "full_text": content}})

    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе: {str(e)}")
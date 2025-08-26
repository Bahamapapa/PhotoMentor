from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import openai
import base64
import io
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_level: str = Form(...),
    detailed: str = Form(...)
):
    print("==> Обработка запроса начата")

    contents = await file.read()
    image_obj = Image.open(io.BytesIO(contents))
    image_obj.thumbnail((786, 786))
    buffered = io.BytesIO()
    image_obj.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = build_prompt(user_level, detailed)

    try:
        print("==> Отправка запроса в OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты — профессиональный визуальный критик и фотограф."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        print("==> Ответ получен")
        content = response['choices'][0]['message']['content']
        print("==> Контент:", content)

        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            parsed = json.loads(json_str)
        except Exception as e:
            print("==> Ошибка разбора JSON:", str(e))
            parsed = {"full_text": content, "regions": []}

        return JSONResponse(content=parsed)

    except Exception as e:
        print("==> КРИТИЧЕСКАЯ ОШИБКА:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


def build_prompt(user_level: str, detailed: str) -> str:
    base = (
        f"Проанализируй художественную фотографию с точки зрения уровня пользователя: {user_level}.\n\n"
        "1. Композиция\n"
        "2. Свет и цвет\n"
        "3. История или эмоция\n"
        "4. Технические параметры\n\n"
        "Сформулируй общее впечатление — короткую оценку работы и дай один совет по улучшению.\n"
        "Оцени фотографию по шкале от 1 до 10 и укажи числовую оценку (например: 'Оценка: 8/10')."
    )

    if detailed.lower() == "true":
        return (
            base
            + "\n\nЗатем выдели 1–5 участков на фото, которые требуют внимания. Для каждого укажи:\n"
            "- координаты (x, y, width, height) в процентах (от 0 до 1);\n"
            "- что именно нужно улучшить.\n\n"
            "Ответ верни строго в формате JSON:\n\n"
            "{\n"
            '  "full_text": "Текст отзыва",\n'
            '  "regions": [\n'
            '    {"x": ..., "y": ..., "width": ..., "height": ..., "comment": "..."}\n'
            "  ]\n"
            "}\n"
            "Никакого другого текста, только JSON."
        )
    else:
        return base
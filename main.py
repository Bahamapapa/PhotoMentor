from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/feedback")
async def feedback(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    # Пока заглушка — вернём фиксированный ответ
    return JSONResponse(content={
        "composition": "Композиция проанализирована...",
        "light_color": "Свет и цвет описаны...",
        "emotion": "Эмоциональный посыл интерпретирован...",
        "technical": "Технические параметры оценены...",
        "score": 7,
        "advice": "Совет: поработай с глубиной кадра."
    })
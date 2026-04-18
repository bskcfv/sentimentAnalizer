from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import pandas as pd
import io

from app.db.connect import get_db
from app.ml.predict import predict_text_batch  # versión batch

router = APIRouter()


@router.post("/post")
async def upload_excel(file: UploadFile = File(...), conn=Depends(get_db)):

    # =========================
    # 1. VALIDAR ARCHIVO
    # =========================
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Formato no soportado")

    content = await file.read()

    # =========================
    # 2. LEER ARCHIVO
    # =========================
    try:
        df = pd.read_excel(io.BytesIO(content))
    except:
        try:
            df = df = pd.read_csv(
                io.BytesIO(content),
                encoding="utf-8"
            )
        except:
            raise HTTPException(status_code=400, detail="Error leyendo el archivo")
    
    
    # =========================
    # 3. NORMALIZAR FORMATO
    # =========================
    if "text" not in df.columns:
        # intenta convertir CSV tipo: "text1, text2, text3"
        if df.shape[1] == 1:
            texts = []
            for row in df.iloc[:, 0]:
                if isinstance(row, str):
                    texts.extend([t.strip() for t in row.split(",") if t.strip()])
            df = pd.DataFrame({"text": texts})
        else:
            raise HTTPException(status_code=400, detail="Debe existir columna 'text'")

    # =========================
    # 4. LIMPIAR DATOS
    # =========================
    texts = df["text"].dropna().astype(str)
    texts = texts[texts.str.strip().str.len() > 0]

    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No hay textos válidos")

    texts_list = texts.tolist()

    # =========================
    # 5. PREDICCIÓN EN BATCH
    # =========================
    predictions = predict_text_batch(texts_list)

    # =========================
    # 6. BULK INSERT (EFICIENTE)
    # =========================
    records = [(text, pred) for text, pred in zip(texts_list, predictions)]

    await conn.executemany("""
        INSERT INTO predictions(text, result)
        VALUES($1, $2)
    """, records)

    # =========================
    # 7. RESPUESTA
    # =========================
    return {
        "processed": len(records),
        "preview": [
            {"text": t, "prediction": p}
            for t, p in zip(texts_list[:10], predictions[:10])
        ]
    }
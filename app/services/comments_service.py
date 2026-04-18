

async def save_prediction(text: str, prediction: str, conn):

    await conn.execute("""
        INSERT INTO predictions(text, result)
        VALUES($1, $2)
    """, text.lower(), prediction)

    return {"prediction": prediction}

async def get_predictions(conn):

    rows = await conn.fetch("SELECT text, result, created_at FROM predictions")
    return [{"text": row["text"], "result": row["result"], "created_at": row["created_at"]} for row in rows]
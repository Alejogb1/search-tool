from fastapi import FastAPI
from api.v1.endpoints import analysis

app = FastAPI(title="Bravo alt API")

app.include_router(analysis.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Bravo alt service is running."}

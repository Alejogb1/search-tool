from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.endpoints import analysis
from data_access.database import SessionLocal
import os

app = FastAPI()

# CORS Configuration
origins = ["*"]  # Adjust for production environments

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Include API routes
app.include_router(analysis.router, prefix="/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Market Intelligence API"}

# Example of how to use the orchestrator (for testing/demonstration)
# In a real application, this would be triggered by an API call.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

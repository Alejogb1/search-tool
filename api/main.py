from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.v1.endpoints import analysis
from data_access.database import SessionLocal
import os
import asyncio
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Market Intelligence API",
    description="API for keyword research and market intelligence",
    version="1.0.0"
)

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

# Add request timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Custom timeout middleware to handle long-running requests"""
    start_time = time.time()

    # Skip timeout for job queuing endpoints (they return immediately)
    job_endpoints = [
        "/v1/keywords/expanded",  # Now uses job queue
        "/v1/keywords/expanded-async",  # Already async
    ]

    if request.url.path in job_endpoints:
        # No timeout for job queuing - they return immediately
        response = await call_next(request)
        elapsed = time.time() - start_time
        if elapsed > 5:  # Log if job queuing takes more than 5 seconds
            logging.warning(f"Slow job queue: {request.method} {request.url.path} took {elapsed:.2f}s")
        return response

    try:
        # Set a timeout for other requests
        response = await asyncio.wait_for(call_next(request), timeout=300.0)

        # Log slow requests
        elapsed = time.time() - start_time
        if elapsed > 60:  # Log requests taking more than 1 minute
            logging.warning(f"Slow request: {request.method} {request.url.path} took {elapsed:.2f}s")

        return response

    except asyncio.TimeoutError:
        logging.error(f"Request timeout: {request.method} {request.url.path} after {time.time() - start_time:.2f}s")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again or use the async endpoint for long-running operations."
        )

# Include API routes
app.include_router(analysis.router, prefix="/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Market Intelligence API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "providers": ["gemini", "openrouter"]}

# Example of how to use the orchestrator (for testing/demonstration)
# In a real application, this would be triggered by an API call.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

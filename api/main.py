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

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check including Redis and database connectivity"""
    import redis
    from data_access.database import SessionLocal

    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {},
        "deployment_info": {
            "platform": "vercel" if os.getenv('KV_URL') else "render",
            "redis_type": "kv" if os.getenv('KV_URL') else "redis"
        }
    }

    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        health_status["services"]["database"] = {"status": "connected"}
        db.close()
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["database"] = {"status": "error", "error": str(e)}

    # Check Redis/KV/Upstash
    try:
        from core.services.job_queue import redis_conn
        ping_start = time.time()
        redis_conn.ping()
        ping_time = time.time() - ping_start

        # Check queue length
        from core.services.job_queue import queue
        queue_length = len(queue)

        # Determine Redis type and configuration
        if os.getenv('UPSTASH_REDIS_REST_URL'):
            redis_type = "upstash"
            redis_url = os.getenv('UPSTASH_REDIS_REST_URL', 'not_set')
        elif os.getenv('KV_URL'):
            redis_type = "kv"
            redis_url = os.getenv('KV_URL', 'not_set')
        else:
            redis_type = "redis"
            redis_url = os.getenv('REDIS_URL', 'not_set')

        health_status["services"]["redis"] = {
            "status": "connected",
            "type": redis_type,
            "ping_time": f"{ping_time:.3f}s",
            "queue_length": queue_length,
            "url_configured": redis_url[:30] + "..." if redis_url != "not_set" else "not_set"
        }

        # Type-specific checks
        if redis_type == "upstash":
            # Test Upstash-specific operations
            test_key = f"health:test:{int(time.time())}"
            redis_conn.set(test_key, "test_value", ex=10)
            test_value = redis_conn.get(test_key)
            health_status["services"]["redis"]["upstash_test"] = test_value == "test_value"

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["redis"] = {
            "status": "error",
            "error": str(e),
            "type": "upstash" if os.getenv('UPSTASH_REDIS_REST_URL') else ("kv" if os.getenv('KV_URL') else "redis")
        }

    # Check environment variables (without exposing sensitive data)
    env_checks = {
        "kv_url": "configured" if os.getenv('KV_URL') else "missing",
        "redis_url": "configured" if os.getenv('REDIS_URL') else "missing",
        "database_url": "configured" if os.getenv('DATABASE_URL') else "missing",
        "google_api_keys": "configured" if os.getenv('GOOGLE_API_KEYS') else "missing"
    }
    health_status["environment"] = env_checks

    # Add troubleshooting tips
    if health_status["status"] == "unhealthy":
        health_status["troubleshooting"] = []

        if health_status["services"].get("redis", {}).get("status") == "error":
            health_status["troubleshooting"].append(
                "Redis connection failed. Check KV_URL/REDIS_URL environment variables in Vercel dashboard."
            )

        if health_status["services"].get("database", {}).get("status") == "error":
            health_status["troubleshooting"].append(
                "Database connection failed. Check DATABASE_URL environment variable."
            )

    return health_status

@app.on_event("startup")
async def startup_checks():
    """Perform startup checks and fail fast if critical services are unavailable"""
    logger.info("🚀 Starting application startup checks...")

    # TEMPORARILY SKIP REDIS CHECK TO ALLOW APP TO START
    # Redis will be checked at runtime when routes are called
    logger.info("ℹ️ Redis check moved to runtime (per-route)")

    # Check database connectivity (this is critical)
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        logger.info("✅ Database connectivity verified on startup")
    except Exception as e:
        logger.error(f"💥 CRITICAL: Database unavailable on startup: {e}")
        logger.error("💥 Application startup failed - Database is required")
        # In production, you might want to exit here
        # import sys; sys.exit(1)

    logger.info("🎉 Application startup checks completed")

# Example of how to use the orchestrator (for testing/demonstration)
# In a real application, this would be triggered by an API call.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

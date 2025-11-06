import redis
import rq
import os
import logging
import time
from fastapi import HTTPException
from core.services.analysis_orchestrator import run_keyword_workflow
from integrations.email_service import email_service

logger = logging.getLogger(__name__)

# Upstash Redis connection for Render deployment
def create_upstash_redis_connection():
    """Create Upstash Redis connection optimized for Render - uses standard Redis client for RQ compatibility"""
    # Always use REDIS_URL with standard Redis client for RQ compatibility
    # RQ internally uses Redis pipelines which don't work with REST clients
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

    logger.info(f"🔗 Connecting to Redis at: {redis_url[:50]}...")

    try:
        # Use standard Redis client for RQ compatibility
        connection_kwargs = {
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'decode_responses': False,  # Disable UTF-8 decoding for compatibility
            'health_check_interval': 30,
        }

        redis_conn = redis.from_url(redis_url, **connection_kwargs)
        redis_conn.ping()
        logger.info("✅ Redis connection established successfully")
        return redis_conn

    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Redis connection failed: {str(e)}. Check REDIS_URL configuration."
        )

# Create optimized Redis connection
redis_conn = create_upstash_redis_connection()
queue = rq.Queue('keyword_expansion', connection=redis_conn)

def background_keyword_expansion(domain: str, email: str = None):
    """Background task to run keyword expansion and send email"""
    job_id = None
    try:
        # Get current job ID for status tracking
        import rq
        job = rq.get_current_job()
        job_id = job.id if job else "unknown"

        logger.info(f"🚀 Starting background job {job_id} for domain: {domain}")

        # Update job status to running
        update_job_status(job_id, "running", "Processing keyword expansion...")

        # Run the keyword workflow
        logger.info(f"📊 Running keyword workflow for {domain}")
        csv_path = run_keyword_workflow(domain)

        # Verify CSV was created
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV file not created at expected path: {csv_path}"
            logger.error(f"❌ {error_msg}")
            update_job_status(job_id, "failed", error_msg)
            raise FileNotFoundError(error_msg)

        # Get file size for logging
        file_size = os.path.getsize(csv_path)
        logger.info(f"✅ Keyword workflow completed successfully. CSV: {csv_path} ({file_size} bytes)")

        # Send email if provided
        if email:
            logger.info(f"📧 Sending email to {email}")
            update_job_status(job_id, "running", "Sending email notification...")
            email_sent = email_service.send_csv_email(email, csv_path, domain)
            if email_sent:
                logger.info("✅ Email sent successfully")
            else:
                logger.warning("⚠️ Email sending failed")
        else:
            logger.info("📧 No email address provided, skipping email notification")

        # Update job status to completed
        update_job_status(job_id, "finished", f"Successfully generated {file_size} bytes CSV")

        logger.info(f"🎉 Background job {job_id} completed successfully")
        return csv_path

    except Exception as e:
        error_msg = f"Background job failed: {str(e)}"
        logger.error(f"💥 {error_msg}", exc_info=True)

        # Update job status to failed
        if job_id:
            update_job_status(job_id, "failed", error_msg)

        raise

def update_job_status(job_id: str, status: str, message: str):
    """Update job status in Redis for monitoring"""
    try:
        # Store status in Redis hash for the job
        status_key = f"job:{job_id}:status"
        redis_conn.hset(status_key, mapping={
            "status": status,
            "message": message,
            "timestamp": str(int(time.time()))
        })
        # Set expiry for status (24 hours)
        redis_conn.expire(status_key, 86400)
        logger.debug(f"Updated job {job_id} status: {status} - {message}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def get_job_status(job_id: str):
    """Get job status from Redis"""
    try:
        # First check RQ job status
        job = queue.fetch_job(job_id)
        if job:
            rq_status = job.get_status()
            rq_result = job.result
            rq_error = str(job.exc_info) if job.exc_info else None

            # Also check our custom status
            status_key = f"job:{job_id}:status"
            custom_status = redis_conn.hgetall(status_key)

            return {
                "job_id": job_id,
                "rq_status": rq_status,
                "rq_result": rq_result,
                "rq_error": rq_error,
                "custom_status": custom_status.decode('utf-8') if custom_status else None,
                "last_updated": custom_status.get(b'timestamp', b'').decode('utf-8') if custom_status else None
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching job status: {e}")
        return None

def queue_keyword_expansion(domain: str, email: str = None):
    """Queue a keyword expansion job"""
    logger.info(f"🔄 Queueing job for domain: {domain}, email: {email}")

    try:
        # Verify Redis connection before queueing
        try:
            redis_conn.ping()
            logger.info("✅ Redis connection verified")
        except Exception as redis_error:
            logger.error(f"❌ Redis connection failed: {redis_error}")
            raise HTTPException(
                status_code=503,
                detail=f"Redis service unavailable: {str(redis_error)}. Please check Redis configuration."
            )

        # Queue the job - RQ handles verification internally
        logger.info("📋 Attempting to enqueue job...")
        job = queue.enqueue(
            background_keyword_expansion,
            domain,
            email,
            job_timeout=3600,  # 1 hour timeout
            result_ttl=86400   # Keep results for 24 hours
        )

        logger.info(f"✅ Job enqueued with ID: {job.id}")

        # Trust RQ's enqueue method - it already verifies job storage
        # No need for manual verification that causes UTF-8 decoding errors
        logger.info(f"🎉 Job {job.id} successfully queued and verified by RQ")
        return job.id

    except HTTPException:
        raise  # Re-raise our custom exceptions
    except Exception as e:
        logger.error(f"💥 CRITICAL: Job queuing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Job queuing failed: {str(e)}. Check Redis configuration and connectivity."
        )

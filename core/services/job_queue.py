import redis
import rq
import os
import logging
import time
from fastapi import HTTPException
from core.services.analysis_orchestrator import run_keyword_workflow
from integrations.email_service import email_service

logger = logging.getLogger(__name__)

# Redis connection using Redis TLS endpoint (works with Upstash, Railway, etc.)
def create_redis_connection():
    """Create Redis connection using standard Redis protocol (TLS)"""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

    logger.info(f"🔗 Connecting to Redis at: {redis_url[:50]}...")

    try:
        # Use standard Redis client - works with Redis TLS endpoints
        redis_conn = redis.from_url(redis_url)
        redis_conn.ping()
        logger.info("✅ Redis connection established successfully")
        return redis_conn

    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Redis connection failed: {str(e)}. Check REDIS_URL configuration."
        )

# Create Redis connection
redis_conn = create_redis_connection()
queue = rq.Queue('keyword_expansion', connection=redis_conn)

def background_keyword_expansion(domain: str, email: str = None):
    """Background task to run keyword expansion and send email"""
    import asyncio

    job_id = None
    try:
        # Get current job ID for status tracking
        import rq
        job = rq.get_current_job()
        job_id = job.id if job else "unknown"

        logger.info(f"🚀 Starting background job {job_id} for domain: {domain}")

        # Update job status to running
        update_job_status(job_id, "running", "Processing keyword expansion...")

        # Run the async keyword workflow in a new event loop
        logger.info(f"📊 Running keyword workflow for {domain}")
        try:
            # Create new event loop for async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            csv_path = loop.run_until_complete(run_keyword_workflow(domain))
            loop.close()
        except Exception as workflow_error:
            logger.error(f"❌ Keyword workflow failed: {workflow_error}")
            update_job_status(job_id, "failed", f"Workflow failed: {str(workflow_error)}")
            raise

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
    logger.info(f"🔍 Checking status for job: {job_id}")
    try:
        # First check active/completed RQ jobs
        job = queue.fetch_job(job_id)
        if job:
            logger.info(f"✅ Active RQ job found: {job_id}")
            rq_status = job.get_status()
            rq_result = job.result
            rq_error = str(job.exc_info) if job.exc_info else None

            # Also check our custom status
            status_key = f"job:{job_id}:status"
            custom_status = redis_conn.hgetall(status_key)

            result = {
                "job_id": job_id,
                "rq_status": rq_status,
                "rq_result": rq_result,
                "rq_error": rq_error,
                "custom_status": custom_status.decode('utf-8') if custom_status else None,
                "last_updated": custom_status.get(b'timestamp', b'').decode('utf-8') if custom_status else None
            }
            logger.info(f"📊 Job {job_id} status: {rq_status}")
            return result

        # If not found in active jobs, check failed jobs
        logger.info(f"🔍 Job {job_id} not in active queue, checking failed jobs...")
        try:
            failed_registry = queue.failed_job_registry
            failed_job = failed_registry.get_job(job_id)
            if failed_job:
                logger.info(f"✅ Failed job found: {job_id}")

                # Get failure details
                exc_string = str(failed_job.exc_info) if failed_job.exc_info else "Unknown error"

                # Also check our custom status
                status_key = f"job:{job_id}:status"
                custom_status = redis_conn.hgetall(status_key)

                result = {
                    "job_id": job_id,
                    "rq_status": "failed",
                    "rq_result": None,
                    "rq_error": exc_string,
                    "custom_status": custom_status.decode('utf-8') if custom_status else "Job failed during execution",
                    "last_updated": custom_status.get(b'timestamp', b'').decode('utf-8') if custom_status else None
                }
                logger.info(f"📊 Failed job {job_id} details retrieved")
                return result
        except Exception as failed_check_error:
            logger.warning(f"⚠️ Error checking failed jobs for {job_id}: {failed_check_error}")

        logger.warning(f"❌ Job {job_id} not found in active or failed queues")
        return None

    except Exception as e:
        logger.error(f"💥 Error fetching job status for {job_id}: {e}", exc_info=True)
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
            result_ttl=86400,  # Keep results for 24 hours
            ttl=86400          # Keep job in queue for 24 hours
        )

        logger.info(f"✅ Job enqueued with ID: {job.id}")

        # Verify job was actually stored
        try:
            stored_job = queue.fetch_job(job.id)
            if stored_job:
                logger.info(f"✅ Job {job.id} verified in queue")
            else:
                logger.error(f"❌ Job {job.id} not found immediately after enqueue!")
                raise HTTPException(500, f"Job storage failed - job not found after enqueue")
        except Exception as verify_error:
            logger.error(f"❌ Job verification failed: {verify_error}")
            raise HTTPException(500, f"Job verification failed: {str(verify_error)}")

        logger.info(f"🎉 Job {job.id} successfully queued and verified")
        return job.id

    except HTTPException:
        raise  # Re-raise our custom exceptions
    except Exception as e:
        logger.error(f"💥 CRITICAL: Job queuing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Job queuing failed: {str(e)}. Check Redis configuration and connectivity."
        )

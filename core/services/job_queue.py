import redis
import rq
import os
import logging
import time
import json
from datetime import datetime, timedelta
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

# Job persistence keys
JOB_PERSISTENCE_KEY = "job_persistence"
WORKER_HEALTH_KEY = "worker_health"
JOB_CHECKPOINT_KEY = "job_checkpoint"

def persist_job_state(job_id: str, domain: str, email: str = None, status: str = "queued", progress: dict = None):
    """Persist job state to Redis for recovery after worker restarts"""
    try:
        job_data = {
            "job_id": job_id,
            "domain": domain,
            "email": email,
            "status": status,
            "created_at": datetime.now().isoformat(),
            "progress": progress or {},
            "last_updated": datetime.now().isoformat()
        }

        redis_conn.hset(JOB_PERSISTENCE_KEY, job_id, json.dumps(job_data))
        redis_conn.expire(JOB_PERSISTENCE_KEY, 604800)  # 7 days
        logger.debug(f"Persisted job state for {job_id}")
    except Exception as e:
        logger.error(f"Failed to persist job state for {job_id}: {e}")

def get_persisted_job(job_id: str):
    """Retrieve persisted job state"""
    try:
        job_data = redis_conn.hget(JOB_PERSISTENCE_KEY, job_id)
        if job_data:
            return json.loads(job_data.decode('utf-8'))
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve persisted job {job_id}: {e}")
        return None

def update_worker_health(worker_id: str, status: str = "alive", current_job: str = None):
    """Update worker health status for monitoring"""
    try:
        health_data = {
            "worker_id": worker_id,
            "status": status,
            "current_job": current_job,
            "last_heartbeat": datetime.now().isoformat(),
            "pid": os.getpid(),
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
        }

        redis_conn.hset(WORKER_HEALTH_KEY, worker_id, json.dumps(health_data))
        redis_conn.expire(WORKER_HEALTH_KEY, 300)  # 5 minutes expiry
        logger.debug(f"Updated worker health for {worker_id}")
    except Exception as e:
        logger.error(f"Failed to update worker health for {worker_id}: {e}")

def get_worker_health():
    """Get all worker health statuses"""
    try:
        workers = redis_conn.hgetall(WORKER_HEALTH_KEY)
        return {k.decode('utf-8'): json.loads(v.decode('utf-8')) for k, v in workers.items()}
    except Exception as e:
        logger.error(f"Failed to get worker health: {e}")
        return {}

def checkpoint_job_progress(job_id: str, progress_data: dict):
    """Save job progress checkpoint for resumption"""
    try:
        checkpoint_key = f"{JOB_CHECKPOINT_KEY}:{job_id}"
        redis_conn.set(checkpoint_key, json.dumps(progress_data), ex=3600)  # 1 hour expiry
        logger.debug(f"Saved checkpoint for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for job {job_id}: {e}")

def get_job_checkpoint(job_id: str):
    """Retrieve job progress checkpoint"""
    try:
        checkpoint_key = f"{JOB_CHECKPOINT_KEY}:{job_id}"
        checkpoint_data = redis_conn.get(checkpoint_key)
        if checkpoint_data:
            return json.loads(checkpoint_data.decode('utf-8'))
        return None
    except Exception as e:
        logger.error(f"Failed to get checkpoint for job {job_id}: {e}")
        return None

def recover_failed_jobs():
    """Recover jobs that were running when worker crashed"""
    try:
        logger.info("🔄 Checking for jobs to recover...")

        # Get all persisted jobs
        persisted_jobs = redis_conn.hgetall(JOB_PERSISTENCE_KEY)
        recovered_count = 0

        for job_id_bytes, job_data_bytes in persisted_jobs.items():
            try:
                job_id = job_id_bytes.decode('utf-8')
                job_data = json.loads(job_data_bytes.decode('utf-8'))

                # Check if job was running and might need recovery
                if job_data.get('status') in ['running', 'started']:
                    # Check if RQ still has the job
                    rq_job = queue.fetch_job(job_id)
                    if not rq_job or rq_job.get_status() in ['failed', None]:
                        logger.info(f"🔄 Recovering job {job_id} that was in status: {job_data.get('status')}")

                        # Re-queue the job
                        new_job = queue.enqueue(
                            background_keyword_expansion,
                            job_data['domain'],
                            job_data.get('email'),
                            job_timeout=3600,
                            result_ttl=86400,
                            ttl=86400
                        )

                        # Update persisted state
                        persist_job_state(new_job.id, job_data['domain'], job_data.get('email'), 'queued')
                        recovered_count += 1

                        logger.info(f"✅ Recovered job {job_id} as new job {new_job.id}")

            except Exception as job_error:
                logger.error(f"Error recovering job {job_id_bytes}: {job_error}")

        if recovered_count > 0:
            logger.info(f"🎉 Recovered {recovered_count} jobs")
        else:
            logger.info("✅ No jobs needed recovery")

    except Exception as e:
        logger.error(f"Failed to recover jobs: {e}")

def background_keyword_expansion(domain: str, email: str = None):
    """Background task to run keyword expansion and send email with persistence and recovery"""
    import asyncio

    job_id = None
    worker_id = f"worker_{os.getpid()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        # Get current job ID for status tracking
        import rq
        job = rq.get_current_job()
        job_id = job.id if job else "unknown"

        logger.info(f"🚀 Starting background job {job_id} for domain: {domain}")

        # Update worker health and persist job state
        update_worker_health(worker_id, "busy", job_id)
        persist_job_state(job_id, domain, email, "running", {"stage": "initializing"})

        # Update job status to running
        update_job_status(job_id, "running", "Processing keyword expansion...")

        # Check for existing checkpoint to resume
        checkpoint = get_job_checkpoint(job_id)
        if checkpoint:
            logger.info(f"📋 Found checkpoint for job {job_id}, resuming from: {checkpoint.get('stage', 'unknown')}")
            # Resume logic would go here - for now, we'll restart
            persist_job_state(job_id, domain, email, "running", {"stage": "resuming", "checkpoint": checkpoint})

        # Run the async keyword workflow in a new event loop
        logger.info(f"📊 Running keyword workflow for {domain}")
        checkpoint_job_progress(job_id, {"stage": "workflow_running", "domain": domain})

        try:
            # Create new event loop for async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            csv_path = loop.run_until_complete(run_keyword_workflow(domain))
            loop.close()
        except Exception as workflow_error:
            logger.error(f"❌ Keyword workflow failed: {workflow_error}")
            persist_job_state(job_id, domain, email, "failed", {"error": str(workflow_error)})
            update_job_status(job_id, "failed", f"Workflow failed: {str(workflow_error)}")
            update_worker_health(worker_id, "idle")
            raise

        # Verify CSV was created
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV file not created at expected path: {csv_path}"
            logger.error(f"❌ {error_msg}")
            persist_job_state(job_id, domain, email, "failed", {"error": error_msg})
            update_job_status(job_id, "failed", error_msg)
            update_worker_health(worker_id, "idle")
            raise FileNotFoundError(error_msg)

        # Get file size for logging
        file_size = os.path.getsize(csv_path)
        logger.info(f"✅ Keyword workflow completed successfully. CSV: {csv_path} ({file_size} bytes)")

        # Update checkpoint
        checkpoint_job_progress(job_id, {"stage": "workflow_complete", "csv_path": csv_path, "file_size": file_size})

        # Send email if provided
        if email:
            logger.info(f"📧 Sending email to {email}")
            update_job_status(job_id, "running", "Sending email notification...")
            checkpoint_job_progress(job_id, {"stage": "sending_email", "email": email})

            email_sent = email_service.send_csv_email(email, csv_path, domain)
            if email_sent:
                logger.info("✅ Email sent successfully")
            else:
                logger.warning("⚠️ Email sending failed")
        else:
            logger.info("📧 No email address provided, skipping email notification")

        # Update job status to completed
        update_job_status(job_id, "finished", f"Successfully generated {file_size} bytes CSV")
        persist_job_state(job_id, domain, email, "finished", {"csv_path": csv_path, "file_size": file_size})

        # Clean up checkpoint and update worker status
        redis_conn.delete(f"{JOB_CHECKPOINT_KEY}:{job_id}")
        update_worker_health(worker_id, "idle")

        logger.info(f"🎉 Background job {job_id} completed successfully")
        return csv_path

    except Exception as e:
        error_msg = f"Background job failed: {str(e)}"
        logger.error(f"💥 {error_msg}", exc_info=True)

        # Update job status to failed and persist state
        if job_id:
            update_job_status(job_id, "failed", error_msg)
            persist_job_state(job_id, domain, email, "failed", {"error": error_msg})

        # Update worker status
        update_worker_health(worker_id, "error")

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

        # Persist job state for recovery
        persist_job_state(job.id, domain, email, "queued")

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

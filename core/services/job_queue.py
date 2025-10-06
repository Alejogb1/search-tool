import redis
import rq
import os
import logging
from core.services.analysis_orchestrator import run_keyword_workflow
from integrations.email_service import email_service

logger = logging.getLogger(__name__)

# Redis connection
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = redis.from_url(redis_url)
queue = rq.Queue('keyword_expansion', connection=redis_conn)

def background_keyword_expansion(domain: str, email: str = None):
    """Background task to run keyword expansion and send email"""
    try:
        logger.info(f"Starting background job for domain: {domain}")

        # Run the keyword workflow
        csv_path = run_keyword_workflow(domain)

        # Send email if provided
        if email and csv_path:
            logger.info(f"Sending email to {email}")
            email_sent = email_service.send_csv_email(email, csv_path, domain)
            if email_sent:
                logger.info("Email sent successfully")
            else:
                logger.warning("Email sending failed")

        return csv_path

    except Exception as e:
        logger.error(f"Background job failed: {e}", exc_info=True)
        raise

def get_job_status(job_id: str):
    """Get job status from Redis"""
    try:
        job = queue.fetch_job(job_id)
        if job:
            return {
                "job_id": job_id,
                "status": job.get_status(),
                "result": job.result,
                "error": str(job.exc_info) if job.exc_info else None
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching job status: {e}")
        return None

def queue_keyword_expansion(domain: str, email: str = None):
    """Queue a keyword expansion job"""
    try:
        job = queue.enqueue(
            background_keyword_expansion,
            domain,
            email,
            job_timeout=3600  # 1 hour timeout
        )
        return job.id
    except Exception as e:
        logger.error(f"Error queueing job: {e}")
        raise

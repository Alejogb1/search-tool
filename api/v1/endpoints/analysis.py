from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
import csv
import os
import logging
import uuid
import asyncio
import json
import time
from core.services.analysis_orchestrator import run_keyword_workflow, expand_input_keywords, generate_csv_from_database
from core.services.job_queue import queue_keyword_expansion, get_job_status
from integrations.llm_client import generate_with_multiple_keys
from integrations.email_service import email_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a background job"""
    logger.info(f"🔍 Checking status for job: {job_id}")

    job_status = get_job_status(job_id)
    if job_status is None:
        logger.warning(f"❌ Job not found: {job_id}")
        raise HTTPException(404, "Job not found")

    logger.info(f"✅ Job found: {job_id}, status: {job_status['rq_status']}")

    # Enhanced response with better status information
    response = {
        "job_id": job_status["job_id"],
        "status": job_status["rq_status"],
        "result": job_status["rq_result"],
        "error": job_status["rq_error"],
        "last_updated": job_status["last_updated"],
        "message": job_status["custom_status"] if job_status["custom_status"] else "Job is processing..."
    }

    # Add status interpretation for frontend
    if job_status["rq_status"] == "queued":
        response["status_description"] = "Job is waiting in queue"
    elif job_status["rq_status"] == "started":
        response["status_description"] = "Job is currently running"
    elif job_status["rq_status"] == "finished":
        response["status_description"] = "Job completed successfully"
    elif job_status["rq_status"] == "failed":
        response["status_description"] = "Job failed - check error details"
    else:
        response["status_description"] = "Job status unknown"

    logger.info(f"📊 Returning status for job {job_id}: {response['status_description']}")
    return response

@router.get("/jobs/{job_id}/download", response_class=FileResponse)
async def download_job_result(job_id: str):
    """Download CSV file for completed job"""
    job_status = get_job_status(job_id)
    if job_status is None:
        raise HTTPException(404, "Job not found")

    if job_status["status"] != "finished":
        raise HTTPException(400, f"Job not completed yet. Status: {job_status['status']}")

    csv_path = job_status.get("result")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(404, "CSV file not found")

    # Extract domain name for filename (we'll need to store this in the job)
    filename = f"keywords-expanded-{job_id}.csv"

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=filename
    )

@router.get("/domains/{domain}/status", response_class=JSONResponse)
async def get_domain_status(domain: str):
    """Check if domain already has processed keywords and return CSV if available"""
    try:
        # Check database for existing keywords
        from data_access.database import SessionLocal
        from data_access import models

        db = SessionLocal()
        try:
            # Get domain record
            db_domain = db.query(models.Domain).filter(models.Domain.url == domain).first()
            if not db_domain:
                return {
                    "domain": domain,
                    "status": "not_found",
                    "message": "Domain not found in database. Needs processing.",
                    "keywords_count": 0,
                    "csv_available": False
                }

            # Get keyword counts by type
            total_keywords = db.query(models.Keyword).filter(models.Keyword.domain_id == db_domain.id).count()
            expanded_keywords = db.query(models.Keyword).filter(
                models.Keyword.domain_id == db_domain.id,
                models.Keyword.keyword_type.in_([models.KeywordType.EXPANDED_ONLY, models.KeywordType.SEED_EXPANDED])
            ).count()

            # Check if CSV file exists
            domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
            csv_filename = f"output-keywords-{domain_name}.csv"
            csv_path = os.path.abspath(csv_filename)
            csv_available = os.path.exists(csv_path)

            if csv_available:
                file_size = os.path.getsize(csv_path)
                return {
                    "domain": domain,
                    "status": "completed",
                    "message": f"Domain fully processed with {total_keywords} keywords",
                    "keywords_count": total_keywords,
                    "expanded_keywords_count": expanded_keywords,
                    "csv_available": True,
                    "csv_path": csv_path,
                    "file_size": file_size,
                    "download_url": f"/domains/{domain}/download"
                }
            elif total_keywords > 100000:
                return {
                    "domain": domain,
                    "status": "ready_for_csv",
                    "message": f"Domain has {total_keywords} keywords in database. CSV can be generated.",
                    "keywords_count": total_keywords,
                    "expanded_keywords_count": expanded_keywords,
                    "csv_available": False,
                    "note": "CSV will be generated on-demand",
                    "generate_csv_url": f"/domains/{domain}/generate-csv"
                }
            else:
                return {
                    "domain": domain,
                    "status": "needs_processing",
                    "message": f"Domain has {total_keywords} keywords. Needs expansion processing.",
                    "keywords_count": total_keywords,
                    "expanded_keywords_count": expanded_keywords,
                    "csv_available": False
                }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error checking domain status: {e}")
        raise HTTPException(500, f"Error checking domain status: {str(e)}")

@router.post("/domains/{domain}/generate-csv", response_class=FileResponse)
async def generate_csv_for_domain(domain: str):
    """Generate CSV file from existing database keywords for a domain"""
    try:
        from data_access.database import SessionLocal
        from data_access import models

        db = SessionLocal()
        try:
            # Get domain record
            db_domain = db.query(models.Domain).filter(models.Domain.url == domain).first()
            if not db_domain:
                raise HTTPException(404, "Domain not found in database")

            # Check if domain has sufficient keywords
            total_keywords = db.query(models.Keyword).filter(models.Keyword.domain_id == db_domain.id).count()
            if total_keywords < 1000:
                raise HTTPException(400, f"Domain has only {total_keywords} keywords. Needs more processing first.")

            # Generate CSV from database
            domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
            csv_path = generate_csv_from_database(db, db_domain.id, domain_name)

            # Verify file was created
            if not os.path.exists(csv_path):
                raise HTTPException(500, "Failed to generate CSV file")

            filename = f"{domain_name}-keywords-expanded.csv"
            return FileResponse(
                path=csv_path,
                media_type="text/csv",
                filename=filename
            )

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating CSV for domain {domain}: {e}")
        raise HTTPException(500, f"Error generating CSV: {str(e)}")

@router.get("/domains/{domain}/download", response_class=FileResponse)
async def download_domain_csv(domain: str):
    """Download existing CSV file for a domain"""
    try:
        # Check if CSV file exists
        domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
        csv_filename = f"output-keywords-{domain_name}.csv"
        csv_path = os.path.abspath(csv_filename)

        if not os.path.exists(csv_path):
            raise HTTPException(404, "CSV file not found for this domain")

        filename = f"{domain_name}-keywords-expanded.csv"
        return FileResponse(
            path=csv_path,
            media_type="text/csv",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV for domain {domain}: {e}")
        raise HTTPException(500, f"Error downloading CSV: {str(e)}")

@router.post("/keywords/expanded-async", response_class=JSONResponse)
async def generate_expanded_async(
    domain: str = Form(...),
    email: str = Form(None)
):
    """Start background keyword expansion job and return job ID immediately"""
    logger.info(f"Starting async job for domain: {domain}, email: {email}")

    try:
        # Queue the job using RQ
        job_id = queue_keyword_expansion(domain, email)

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Keyword expansion started in background. Use GET /jobs/{job_id} to check status."
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they have proper status codes)
        raise
    except Exception as e:
        logger.error(f"Failed to queue job: {e}")
        raise HTTPException(500, f"Failed to queue job: {str(e)}")

@router.post("/keywords/seeds", response_class=PlainTextResponse)
async def generate_seeds(domain: str):
    """Generate seed keywords from domain content"""
    try:
        api_keys_str = os.getenv("GOOGLE_API_KEYS")
        API_KEYS = [key.strip() for key in api_keys_str.split(',')]
        MODEL_CONFIGS = [
            {"name": "gemini-2.5-pro", "rpm": 5},
            {"name": "gemini-2.5-flash", "rpm": 10},
            {"name": "gemini-2.5-flash-lite-preview-06-17", "rpm": 15},
            {"name": "gemini-2.0-flash", "rpm": 15},
            {"name": "gemini-2.0-flash-lite", "rpm": 30},
        ]

        # Generate seed keywords with timeout handling
        seeds = await generate_with_multiple_keys(
            context_source="domain",
            domain_url=domain,
            api_keys=API_KEYS,
            model_configs=MODEL_CONFIGS,
            output_file='input-keywords-DOMAIN-.txt',
            parallel=True,
            mode="commercial"
        )

        # Handle different response types
        if isinstance(seeds, str) and seeds.startswith("Error:"):
            raise HTTPException(500, f"Keyword generation failed: {seeds}")
        elif isinstance(seeds, str):
            return seeds
        else:
            # If it's a set or other collection, convert to string
            return "\n".join(seeds) if seeds else ""

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_seeds endpoint: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@router.post("/keywords/expanded", response_class=JSONResponse)
async def generate_expanded(domain: str = Form(...), email: str = Form(None)):
    """
    Generate expanded keywords CSV with database integration and optional email delivery.

    ⚠️  BREAKING CHANGE: Due to deployment timeout limits, this endpoint now uses
    background job processing instead of synchronous execution.

    Returns job ID immediately. Use GET /jobs/{job_id} to check status and
    GET /jobs/{job_id}/download to get the final CSV when complete.
    """
    logger.info(f"Endpoint /keywords/expanded called with domain: {domain}, email: {email}")

    try:
        logger.info("Queueing keyword expansion job for background processing...")

        # Queue the job using RQ (same as the async endpoint)
        job_id = queue_keyword_expansion(domain, email)

        logger.info(f"Job queued successfully with ID: {job_id}")

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Keyword expansion job started in background. This process takes 5-15 minutes.",
            "instructions": {
                "check_status": f"GET /jobs/{job_id}",
                "download_csv": f"GET /jobs/{job_id}/download (when status is 'finished')",
                "email_notification": f"CSV will be emailed to {email}" if email else "No email specified"
            },
            "estimated_completion": "5-15 minutes",
            "note": "This endpoint now uses background processing due to deployment timeout limits."
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is (they have proper status codes)
        raise
    except Exception as e:
        logger.error(f"Failed to queue keyword expansion job: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to start keyword expansion job: {str(e)}")

# Add route aliases to match frontend expectations
@router.post("/api/v1/analyze", response_class=JSONResponse)
async def analyze_endpoint(domain: str = Form(...), email: str = Form(None)):
    """
    Alias for /keywords/expanded to match frontend expectations.
    Frontend expects POST /api/v1/analyze but backend provides /v1/keywords/expanded
    """
    logger.info(f"Alias endpoint /api/v1/analyze called with domain: {domain}, email: {email}")
    return await generate_expanded(domain, email)

@router.get("/api/v1/report/{job_id}", response_class=JSONResponse)
async def report_endpoint(job_id: str):
    """
    Alias for /jobs/{job_id} to match frontend expectations.
    Frontend expects GET /api/v1/report/[job_id] but backend provides /v1/jobs/[job_id]
    """
    logger.info(f"Alias endpoint /api/v1/report/{job_id} called")
    return await get_job_status_endpoint(job_id)

@router.post("/keywords/expand-input", response_class=PlainTextResponse)
async def expand_input_keywords_endpoint(input_file: str = "input-keywords.txt"):
    """Expand existing keywords from input file via Google Ads API"""
    try:
        expanded_keywords = await expand_input_keywords(input_file)
        # Return expanded keywords as newline-separated text
        return "\n".join([kw.get('text', '') for kw in expanded_keywords])
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/debug/redis", response_class=JSONResponse)
async def debug_redis():
    """Debug endpoint to test Redis connectivity"""
    try:
        from core.services.job_queue import redis_conn

        # Test basic Redis operations
        redis_conn.ping()

        # Get queue info
        queue_length = len(queue)
        queue_name = queue.name

        # Test job storage
        test_key = f"debug:test:{int(time.time())}"
        redis_conn.set(test_key, "test_value", ex=60)  # Expires in 60 seconds
        test_value = redis_conn.get(test_key)

        return {
            "redis_status": "connected",
            "queue_name": queue_name,
            "queue_length": queue_length,
            "test_key_set": test_value == "test_value",
            "redis_url": os.getenv('REDIS_URL', 'not_set')[:20] + "..." if os.getenv('REDIS_URL') else "not_set"
        }

    except Exception as e:
        logger.error(f"Redis debug failed: {e}")
        return {
            "redis_status": "error",
            "error": str(e),
            "redis_url": os.getenv('REDIS_URL', 'not_set')
        }

@router.get("/debug/jobs", response_class=JSONResponse)
async def debug_jobs():
    """Debug endpoint to list all jobs in queue"""
    try:
        from core.services.job_queue import queue

        # Get all jobs in queue
        jobs = queue.get_jobs()

        job_list = []
        for job in jobs:
            job_info = {
                "id": job.id,
                "status": job.get_status(),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "args": job.args if hasattr(job, 'args') else None,
                "result": job.result if hasattr(job, 'result') else None,
                "error": str(job.exc_info) if hasattr(job, 'exc_info') and job.exc_info else None
            }
            job_list.append(job_info)

        return {
            "queue_name": queue.name,
            "total_jobs": len(job_list),
            "jobs": job_list
        }

    except Exception as e:
        logger.error(f"Jobs debug failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/debug/workers", response_class=JSONResponse)
async def debug_workers():
    """Debug endpoint to check worker health status"""
    try:
        from core.services.job_queue import get_worker_health

        workers = get_worker_health()

        return {
            "total_workers": len(workers),
            "workers": workers,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Workers debug failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/debug/persistence", response_class=JSONResponse)
async def debug_persistence():
    """Debug endpoint to check job persistence state"""
    try:
        from core.services.job_queue import redis_conn, JOB_PERSISTENCE_KEY

        # Get all persisted jobs
        persisted_jobs = redis_conn.hgetall(JOB_PERSISTENCE_KEY)

        jobs_data = {}
        for job_id_bytes, job_data_bytes in persisted_jobs.items():
            try:
                job_id = job_id_bytes.decode('utf-8')
                job_data = json.loads(job_data_bytes.decode('utf-8'))
                jobs_data[job_id] = job_data
            except Exception as parse_error:
                jobs_data[job_id_bytes.decode('utf-8')] = f"Parse error: {parse_error}"

        return {
            "total_persisted_jobs": len(jobs_data),
            "persisted_jobs": jobs_data,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Persistence debug failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/debug/recover-jobs", response_class=JSONResponse)
async def debug_recover_jobs():
    """Debug endpoint to manually trigger job recovery"""
    try:
        from core.services.job_queue import recover_failed_jobs

        logger.info("🔄 Manual job recovery triggered via debug endpoint")
        recover_failed_jobs()

        return {
            "status": "success",
            "message": "Job recovery completed",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Manual job recovery failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

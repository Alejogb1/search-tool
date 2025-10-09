from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
import csv
import os
import logging
import uuid
import asyncio
from core.services.analysis_orchestrator import run_keyword_workflow, expand_input_keywords
from core.services.job_queue import queue_keyword_expansion, get_job_status
from integrations.llm_client import generate_with_multiple_keys
from integrations.email_service import email_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a background job"""
    job_status = get_job_status(job_id)
    if job_status is None:
        raise HTTPException(404, "Job not found")

    return job_status

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

    except Exception as e:
        logger.error(f"Failed to queue keyword expansion job: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to start keyword expansion job: {str(e)}")

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

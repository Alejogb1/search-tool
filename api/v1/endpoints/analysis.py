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
        

        # Generate seed keywords
        seeds = await generate_with_multiple_keys( context_source="domain",
        domain_url=domain,
        api_keys=API_KEYS, 
        model_configs=MODEL_CONFIGS, 
        output_file='input-keywords-DOMAIN-.txt', 
        parallel=True, 
        mode="commercial")
        return "\n".join(seeds)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/keywords/expanded", response_class=FileResponse)
async def generate_expanded(domain: str = Form(...), email: str = Form(None)):
    """Generate and return expanded keywords CSV with database integration and optional email delivery"""
    logger.info(f"Endpoint /keywords/expanded called with domain: {domain}, email: {email}")
    csv_path = None  # Initialize csv_path
    try:
        logger.info("Starting keyword workflow execution...")
        csv_path = await run_keyword_workflow(domain)
        logger.info(f"Keyword workflow completed successfully. Consolidated CSV path: {csv_path}")

        # Check if file exists and get size
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            logger.info(f"Generated consolidated CSV file size: {file_size} bytes")
        else:
            logger.error(f"Consolidated CSV file was not created at expected path: {csv_path}")
            raise HTTPException(500, "Consolidated CSV file was not generated")

        # Send email if email parameter is provided
        if email:
            logger.info(f"Sending CSV via email to: {email}")
            email_sent = email_service.send_csv_email(email, csv_path, domain)
            if email_sent:
                logger.info(f"CSV email sent successfully to {email}")
            else:
                logger.warning(f"Failed to send CSV email to {email}, but continuing with file response")

        # Extract domain name for filename
        domain_name = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('.')[0].upper()
        filename = f"{domain_name}-keywords-expanded.csv"

        logger.info(f"Returning FileResponse with filename: {filename}")
        return FileResponse(
            path=csv_path,
            media_type="text/csv",
            filename=filename
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Endpoint /keywords/expanded failed with error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
    finally:
        # Note: Consolidated CSV files are kept for reference, not cleaned up like temp files
        logger.info("Consolidated CSV file retained for reference")

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

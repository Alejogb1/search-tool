from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import csv
import os
from core.services.analysis_orchestrator import run_keyword_workflow
from integrations.llm_client import generate_with_multiple_keys

router = APIRouter()

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
async def generate_expanded(domain: str):
    """Generate and return expanded keywords CSV"""
    csv_path = None  # Initialize csv_path
    try:
        csv_path = await run_keyword_workflow(domain)
        return FileResponse(
            path=csv_path,
            media_type="text/csv",
            filename=f"{domain}-keywords.csv"
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

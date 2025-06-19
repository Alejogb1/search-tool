from fastapi import APIRouter

router = APIRouter()

@router.get("/domain/{domain}")
async def analyze(domain: str):
    url = f"http://{domain}"
    return {"url": url, "domain": domain}


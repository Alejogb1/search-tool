from pydantic import BaseModel, HttpUrl

class AnalysisRequest(BaseModel):
    url: HttpUrl
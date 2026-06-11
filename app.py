from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from analyzer import run_full_analysis
from rag_engine import add_incident
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class LogRequest(BaseModel):
    logs: str


class IncidentRequest(BaseModel):
    id: str
    title: str
    description: str
    root_cause: str
    resolution: str
    service: str
    severity: str
    resolved_in: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/analyze")
async def analyze(req: LogRequest):
    """Main analysis endpoint."""
    try:
        result = run_full_analysis(req.logs)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/add-incident")
async def add_incident_route(req: IncidentRequest):
    """Add a resolved incident to the RAG knowledge base."""
    try:
        add_incident(req.dict())
        return {"success": True, "message": f"Added {req.id} to knowledge base"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded log file."""
    try:
        content = await file.read()
        logs = content.decode("utf-8", errors="ignore")
        result = run_full_analysis(logs)
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
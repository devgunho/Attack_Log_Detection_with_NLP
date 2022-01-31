import os

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from glob import glob
from pydantic import BaseModel

# FastAPI application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SystemLogCreateRequest(BaseModel):
    system_name: str
    log_text: str

@app.post("/logs")
def create_log(log_request: SystemLogCreateRequest):
    # Logic for analyzing system logs
    return {"message": "Log analysis completed"}

@app.get("/")
def main(request: Request):
    scripts_path = "/js/script.js"
    return templates.TemplateResponse("index.html", {"request": request, "scripts_path": scripts_path})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ai_utils import *

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        return f.read()

class Behavior(BaseModel):
    reading_time: int
    pause_time: int
    rereads: int

class Face(BaseModel):
    blink_rate: int
    eyebrow_movements: int
    head_movements: int

class InputData(BaseModel):
    behavior: Behavior
    face: Face
    content: str

@app.post("/analyze")
def analyze(data: InputData):
    load = compute_cognitive_load(
        data.behavior.dict(),
        data.face.dict()
    )

    temperature, pace = map_parameters(load)
    script = generate_script(data.content, temperature)
    scenes = generate_scenes(script)

    return {
        "cognitive_load": load,
        "temperature": temperature,
        "pace": pace,
        "script": script,
        "scenes": scenes
    }

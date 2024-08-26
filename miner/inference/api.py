from contextlib import asynccontextmanager
from io import BytesIO

from PIL.Image import MIME
from fastapi import FastAPI, Request
from starlette.responses import Response

from pipeline import load_pipeline, GenerationRequest, infer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline = load_pipeline()

    yield {"pipeline": pipeline}


app = FastAPI(lifespan=lifespan)


@app.post("/")
def generate(body: GenerationRequest, request: Request):
    image = infer(body, request.app.state["pipeline"])

    data = BytesIO()
    image.save(data, format=image.format)

    return Response(data, media_type=MIME[image.format])

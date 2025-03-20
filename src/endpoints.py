# External imports
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Request
import fastapi
import uvicorn

# Internal imports
from src.logger.logger import get_logger
from src.cleanup import (
    user_pipelines,
    get_user_pipeline,
    mask_ip,
    lifespan,
)

logger = get_logger(__name__)

app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/generate")
async def generate_image(prompt: str, request: Request):
    """
    Generate an image using a user-specific pipeline instance.
    Each client gets their own pipeline, which is reused across their requests.
    Inactive pipelines are automatically cleaned up in the background.
    """
    # Return if prompt is empty
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Get full IP and log the masked version
    client_ip = request.client.host
    masked_ip = mask_ip(client_ip)
    logger.info(f"Request from {masked_ip}. Prompt: {prompt}")

    try:
        sd = get_user_pipeline(client_ip)
        image = sd.generate_image(prompt)
        b64_image = sd.convert_into_base64(image)
    except Exception as e:
        # If there's an error, clean up this user's pipeline
        if client_ip in user_pipelines:
            sd, _ = user_pipelines[client_ip]
            sd.cleanup()
            del user_pipelines[client_ip]
        raise HTTPException(
            status_code=500, detail=f"Failed to generate image: {str(e)}"
        )

    logger.info(
        f"Image generated successfully with size of {len(b64_image)/1000} kb. Returning to client.."
    )
    return {"image": b64_image}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

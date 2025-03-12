import fastapi
from fastapi.middleware.cors import CORSMiddleware
from src.stable_diffusion import StableDiffusion
import uvicorn
from fastapi import HTTPException, Request
from typing import Dict
import time

app = fastapi.FastAPI()

# Dictionary to store user pipelines and their last activity time
user_pipelines: Dict[str, tuple[StableDiffusion, float]] = {}

# Cleanup threshold in seconds
CLEANUP_THRESHOLD = 3600  # Open for 1 hour during development

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_inactive_pipelines():
    """Remove pipelines that haven't been used for a while"""
    current_time = time.time()
    inactive_users = [
        ip
        for ip, (_, last_active) in user_pipelines.items()
        if current_time - last_active > CLEANUP_THRESHOLD
    ]

    for ip in inactive_users:
        print(f"Cleaning up inactive pipeline for {ip}")
        sd, _ = user_pipelines[ip]
        sd.cleanup()
        del user_pipelines[ip]


def get_user_pipeline(client_ip: str) -> StableDiffusion:
    """Get or create a pipeline for the user"""
    current_time = time.time()

    # Clean up old pipelines first
    cleanup_inactive_pipelines()

    # Return existing pipeline if available
    if client_ip in user_pipelines:
        sd, _ = user_pipelines[client_ip]
        user_pipelines[client_ip] = (
            sd,
            current_time,
        )  # Update last active time
        return sd

    # Create new pipeline
    try:
        sd = StableDiffusion()
        sd.initialize()
        user_pipelines[client_ip] = (sd, current_time)
        return sd
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize pipeline: {str(e)}"
        )


@app.get("/generate")
async def generate_image(prompt: str, request: Request):
    """
    Generate an image using a user-specific pipeline instance.
    Each client gets their own pipeline, which is reused across their requests.
    Inactive pipelines are automatically cleaned up to free resources.
    """
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    client_ip = request.client.host
    print(f"Request from {client_ip}. Prompt: {prompt}")

    try:
        sd = get_user_pipeline(client_ip)
        image = sd.generate_image(prompt)
        return {"image": sd.convert_into_base64(image)}
    except Exception as e:
        # If there's an error, clean up this user's pipeline
        if client_ip in user_pipelines:
            sd, _ = user_pipelines[client_ip]
            sd.cleanup()
            del user_pipelines[client_ip]
        raise HTTPException(
            status_code=500, detail=f"Failed to generate image: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

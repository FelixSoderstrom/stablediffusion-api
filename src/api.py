import fastapi
from fastapi.middleware.cors import CORSMiddleware
from src.stable_diffusion import StableDiffusion
import uvicorn

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_image")
def generate_image(prompt: str):
    """
    Remove the prints later. They are here for debugging.

    This endpoint is the most basic form of the API.
    It initializes the SD class, generates, converts,
    returns an image then cleans up pipeline.

    Ideally, we want to initialize the SD class
    once and reuse it for multiple requests.
    In a different endpoint maybe.
    """
    print(f"Request received.\nPrompt: {prompt}")
    sd = StableDiffusion()

    print("Initializing Stable Diffusion...")
    sd.initialize()

    print("Generating image...")
    image = sd.generate_image(prompt)

    print("Converting image to base64...")
    base64_image = sd.convert_into_base64(image)

    print("Cleaning up...")
    sd.cleanup()

    print("Returning image...")
    return {"image": base64_image}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

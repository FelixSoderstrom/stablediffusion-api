"""
Run this script to generate a test image without the API/WEBUI.
"""

from src.stable_diffusion import StableDiffusion
import requests
import base64
from PIL import Image
import io
from logger.logger import get_logger

logger = get_logger(__name__)
logger.info("Starting test..")

# Cow31337Killer
prompt = "A knight in blue armor holding a gigantic axe, standing in a cow pen, cows in the background, cinematic lighting, detailed shading, high resolution, detailed background and foreground elements"
url = "http://127.0.0.1:8001/generate"
response = requests.get(url, params={"prompt": prompt})
image = response.json()["image"]
image = Image.open(io.BytesIO(base64.b64decode(image)))
logger.info("Image generated, opening image..")
image.show()

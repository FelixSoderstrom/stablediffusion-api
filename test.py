"""
Run this script to generate a test image without the API/WEBUI.
"""

from src.stable_diffusion import StableDiffusion

# Cow31337Killer
prompt = "A knight in blue armor holding a gigantic axe, standing in a cow pen, cows in the background, cinematic lighting, detailed shading, high resolution, detailed background and foreground elements"

generator = StableDiffusion()

generator.initialize()

image = generator.generate_image(prompt)


image.show()

generator.cleanup()

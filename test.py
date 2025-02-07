"""
Run this to generate a test image.
"""

from stable_diffusion import StableDiffusion
from PIL import Image


prompt = "An image of a ugly woman with long blonde hair and blue eyes sitting at a busstop, cinematic lighting, detailed shawoding, detailed eyes, high resolution, detailed background and foreground elements"

generator = StableDiffusion()

generator.initialize()

image = generator.generate_image(prompt)


image.show()

generator.cleanup()

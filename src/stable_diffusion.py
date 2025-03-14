import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from dotenv import load_dotenv
from io import BytesIO
import base64
from diffusers import DPMSolverMultistepScheduler


class StableDiffusion:
    def __init__(self):
        load_dotenv(override=True)

        # Find the model
        self.models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )
        self.model_name = os.getenv("IMAGE_MODEL")
        self.model_file = f"{self.model_name}.safetensors"

        # Initialize pipeline as None
        self.pipeline = None
        self._is_initialized = False

        # Set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for image generation")
        self.device = "cuda"

        # Config for generation
        self.default_config = {
            "num_inference_steps": 5,
            "guidance_scale": 2,
            "negative_prompt": "text, watermark, logo, title, signature, blurry, low quality, distorted",
        }

    def initialize(self):
        """Initialize the Stable Diffusion pipeline."""
        if self._is_initialized:
            return

        try:
            model_path = os.path.join(self.models_dir, self.model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load and optimize pipeline
            vae = AutoencoderKL.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                vae=vae,
            )

            self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_tiling()

            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # Xformers not available

            # Set DPM++ Multistep as the default scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
            )

            self._is_initialized = True

        except Exception as e:
            self.cleanup()
            raise e

    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_initialized = False

    def generate_image(
        self, prompt: str, width: int = 768, height: int = 768
    ) -> Image.Image:
        """
        Generate an image from a prompt.

        Args:
            prompt (str): The prompt to generate an image from
            width (int): Image width (default: 1024)
            height (int): Image height (default: 1024)

        Returns:
            PIL.Image.Image: The generated image
        """
        if not self._is_initialized:
            self.initialize()

        try:
            # Generate the image
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=self.default_config["negative_prompt"],
                num_inference_steps=self.default_config[
                    "num_inference_steps"
                ],
                guidance_scale=self.default_config["guidance_scale"],
                width=width,
                height=height,
            )

            # Return the first image (should only be one)
            return output.images[0]

        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")

    def convert_into_base64(self, image: Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

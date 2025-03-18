import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from dotenv import load_dotenv
from io import BytesIO
import base64
from diffusers import DPMSolverMultistepScheduler
from src.logger.logger import get_logger
from fastapi import HTTPException


class StableDiffusion:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing Stable Diffusion")
        load_dotenv(override=True)

        # Find the model
        self.models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )
        self.model_name = os.getenv("IMAGE_MODEL")
        self.model_file = f"{self.model_name}.safetensors"
        self.logger.info(f"Using model: {self.model_file}")

        # Initialize pipeline as None
        self.pipeline = None
        self._is_initialized = False

        # Set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for image generation")
        self.device = "cuda"
        self.logger.info(f"Using device: {self.device}")

        # Config for generation
        self.default_config = {
            "num_inference_steps": 5,
            "guidance_scale": 2,
            "negative_prompt": "text, watermark, logo, title, signature, blurry, low quality, distorted",
        }
        self.logger.info("StableDiffusion class has been instantialized.")

    def initialize(self):
        """Initialize the Stable Diffusion pipeline."""
        if self._is_initialized:
            self.logger.info("Pipeline already initialized.")
            return

        try:
            self.logger.info("Initializing new pipeline")

            model_path = os.path.join(self.models_dir, self.model_file)
            if not os.path.exists(model_path):
                self.logger.error(f"Model not found at {model_path}")
                raise FileNotFoundError(f"Model not found at {model_path}")
            self.logger.info(f"Model found")

            # Load and optimize pipeline
            vae = AutoencoderKL.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            self.logger.info("VAE loaded successfully")
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                vae=vae,
            )

            self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_tiling()
            self.logger.info("Pipeline enabled")
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("Xformers enabled")
            except Exception:
                self.logger.warning("Xformers not available")
                pass  # Xformers not available

            # Set DPM++ Multistep as the default scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
            )
            self.logger.info("DPM++ Multistep scheduler enabled")

            self._is_initialized = True
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.cleanup()
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            self.logger.error(f"Pipeline not initialized")
            raise e

    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            self.logger.info("Cleaning up pipeline..")
            del self.pipeline
            self.pipeline = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_initialized = False
            self.logger.info("Pipeline cleaned up")

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
            self.logger.info("Pipeline not initialized, initializing..")
            self.initialize()

        try:
            self.logger.info("Generating image..")
            # Generate the image
            try:
                output: StableDiffusionPipelineOutput = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.default_config["negative_prompt"],
                    num_inference_steps=self.default_config[
                        "num_inference_steps"
                    ],
                    guidance_scale=self.default_config["guidance_scale"],
                    width=width,
                    height=height,
                )
            except Exception as e:
                self.logger.error(f"Error generating image: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating image: {str(e)}",
                )

            self.logger.info("Image generated successfully")
            return output.images[0]

        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")

    def convert_into_base64(self, image: Image.Image):
        """Convert an image into a base64 string."""
        self.logger.info("Converting image into base64..")
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            self.logger.error(f"Error converting image into base64: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error converting image into base64: {str(e)}",
            )

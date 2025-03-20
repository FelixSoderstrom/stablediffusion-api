"""
File for handling cleanup and IP addresses
"""

# External imports
from typing import Dict
import time
from fastapi import HTTPException
import asyncio
import ipaddress
from contextlib import asynccontextmanager
import fastapi


# Internal imports
from src.stable_diffusion import StableDiffusion
from src.logger.logger import get_logger

# Dictionary to store user pipelines and their last activity time
user_pipelines: Dict[str, tuple[StableDiffusion, float]] = {}
CLEANUP_THRESHOLD = 3600  # Open for 1 hour during development
CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes
cleanup_task = None  # Task reference to manage background tasks
logger = get_logger(__name__)


def mask_ip(client_ip: str) -> str:
    """Masks the IP address"""
    try:
        ip = ipaddress.ip_address(client_ip)
        if ip.version == 4:
            network = ipaddress.IPv4Network(f"{ip}/24", strict=False)
            masked_ip = str(network.network_address)
        else:
            network = ipaddress.IPv6Network(f"{ip}/64", strict=False)
            masked_ip = str(network.network_address)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to mask IP")

    return masked_ip


async def cleanup_inactive_pipelines():
    """Remove pipelines that haven't been used for a while"""
    current_time = time.time()
    inactive_users = [
        ip
        for ip, (_, last_active) in user_pipelines.items()
        if current_time - last_active > CLEANUP_THRESHOLD
    ]

    for ip in inactive_users:
        logger.info(f"Cleaning up inactive pipeline for {mask_ip(ip)}")
        sd, _ = user_pipelines[ip]
        sd.cleanup()
        del user_pipelines[ip]


async def periodic_cleanup():
    """Run cleanup task periodically in the background"""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        await cleanup_inactive_pipelines()


def get_user_pipeline(client_ip: str) -> StableDiffusion:
    """Get or create a pipeline for the user"""
    current_time = time.time()

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


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Startup: create the background task
    global cleanup_task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Background cleanup task started")

    yield

    # Shutdown: cancel the task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Background cleanup task cancelled")

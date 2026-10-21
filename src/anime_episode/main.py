"""ASGI entrypoint: uvicorn anime_episode.main:app"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from anime_episode import __version__
from anime_episode.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Anime Episode Generator",
    description="Story → open-source anime episode pipeline (AniSora / Wan2.2 stack).",
    version=__version__,
)
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "anime_episode",
        "version": __version__,
        "docs": "/docs",
        "health": "/v1/health",
    }


def run() -> None:
    import uvicorn

    from anime_episode.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "anime_episode.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()

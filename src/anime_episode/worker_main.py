"""Long-running queue worker: python -m anime_episode.worker_main"""

from __future__ import annotations

import asyncio
import logging

from anime_episode.api.deps import get_queue, get_runner, get_store
from anime_episode.config import get_settings
from anime_episode.queue.worker import Worker

logging.basicConfig(level=logging.INFO)


async def _amain() -> None:
    settings = get_settings()
    worker = Worker(
        queue=get_queue(),
        store=get_store(),
        runner=get_runner(),
        chain=settings.worker_chain,
        concurrency=settings.worker_concurrency,
    )
    await worker.run_forever()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

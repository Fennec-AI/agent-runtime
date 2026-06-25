"""Allows `python -m cli` to invoke cli.main."""
from .main import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())

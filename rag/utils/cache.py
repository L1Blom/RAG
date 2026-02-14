"""Caching utilities for performance optimisation.

Provides simple LRU-cached helpers that can be used by services to avoid
repeated expensive operations (e.g. re-computing the same embedding or
re-reading an unchanged config file).
"""

from functools import lru_cache
from typing import Tuple
import configparser
import logging


@lru_cache(maxsize=128)
def cached_embedding(text: str, model: str) -> Tuple[float, ...]:
    """
    Cache embeddings for frequently used text.

    .. note::
        This is a placeholder.  Actual integration with
        :class:`~rag.services.embeddings.EmbeddingsService` should replace
        the body once hot-path profiling confirms benefit.

    Args:
        text: The text to embed.
        model: The embedding model identifier.

    Returns:
        Tuple of floats representing the embedding vector.
    """
    # Placeholder – will be wired into EmbeddingsService
    raise NotImplementedError("Wire this into EmbeddingsService first")


@lru_cache(maxsize=32)
def cached_config(config_path: str) -> dict:
    """
    Cache the sections/keys of a configuration file.

    Useful when the same file is read many times in a single process
    lifetime (e.g. provider factory asking for the same INI values).

    Args:
        config_path: Absolute or relative path to the INI file.

    Returns:
        A dict of ``{section: {key: value, …}, …}``.
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)
    result = {}
    for section in parser.sections():
        result[section] = dict(parser.items(section))
    logging.debug("Cached config file: %s (%d sections)", config_path, len(result))
    return result


def clear_all_caches() -> None:
    """Clear every LRU cache defined in this module."""
    cached_embedding.cache_clear()
    cached_config.cache_clear()
    logging.debug("All caches cleared")

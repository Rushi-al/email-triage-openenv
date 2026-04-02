"""Compatibility shim. Use server.app instead."""
from server.app import app

__all__ = ["app"]

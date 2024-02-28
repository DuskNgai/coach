from typing import IO

from iopath.common.file_io import (
    PathHandler,
    PathManager,
    HTTPURLHandler,
    OneDrivePathHandler,
)

__all__ = [
    "PathManagerSingleton",
    "PathHandler",
]

PathManagerSingleton = PathManager()
"""A Coach project-specific path manager."""

class CoachHandler(PathHandler):
    """
    Resolve paths to Coach checkpoints, datasets, and other resources.
    """

    PREFIX = "coach://"
    SOURCE_PREFIX = None

    def _get_supported_prefixes(self):
        return [CoachHandler.PREFIX]

    def _get_local_path(self, path: str, **kwargs) -> str:
        """
        Substitutes the prefix of the path with the source prefix.

        Args:
            `path` (str): The path to the resource with the Coach prefix.

        Returns:
            str: The path to the resource with the source prefix.
        """
        name = path[len(CoachHandler.PREFIX):]
        return PathManagerSingleton.get_local_path(CoachHandler.SOURCE_PREFIX + name, **kwargs)

    def _open(self, path: str, mode='r', **kwargs) -> IO[str] | IO[bytes]:
        name = path[len(CoachHandler.PREFIX):]
        return PathManagerSingleton.open(CoachHandler.SOURCE_PREFIX + name, mode, **kwargs)

PathManagerSingleton.register_handler(HTTPURLHandler())
PathManagerSingleton.register_handler(OneDrivePathHandler())
PathManagerSingleton.register_handler(CoachHandler())

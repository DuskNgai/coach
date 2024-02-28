import logging
import os
from typing import Any
from urllib.parse import urlparse

from fvcore.common.checkpoint import _IncompatibleKeys, Checkpointer
import torch
from torch.nn.parallel import DistributedDataParallel

from coach.utils import comm
from coach.utils.file_io import PathManagerSingleton

class CoachCheckpointer(Checkpointer):
    """
    Coach specified checkpointer.
    """

    def __init__(self,
        model: torch.nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables,
    ) -> None:
        super().__init__(
            model=model,
            save_dir=save_dir,
            save_to_disk=comm.is_main_process() if save_to_disk is None else save_to_disk,
            **checkpointables
        )
        self.path_manager = PathManagerSingleton

    def load(self, path: str, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("[CoachCheckpointer] Loading checkpoint from {}.".format(path))

        need_sync = False
        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            this_process_has_file = os.path.isfile(path)
            all_processes_have_file = comm.all_gather(this_process_has_file)

            # Synchornize the model states.
            if not all_processes_have_file[0]:
                raise FileNotFoundError("Checkpoint {} not found on main process!".format(path))
            if not all(all_processes_have_file):
                logger.warning(
                    "Checkpoint {} not found on all processes!".format(path)
                )
                need_sync = True
            if not this_process_has_file:
                path = None

        result = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker.")
            self.model._sync_final_model(this_process_has_file)

        return result

    def _load_file(self, filename: str) -> Any:
        loaded = self._torch_load(filename)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _torch_load(self, filename: str) -> dict[str, Any]:
        return super()._load_file(filename)

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        incompatible = super()._load_model(checkpoint)
        return incompatible

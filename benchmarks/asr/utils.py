from typing import Tuple

from pfl.callback import TrainingProcessCallback
from pfl.metrics import Metrics
from pfl.model.base import ModelType
from pfl.internal.ops.selector import get_default_framework_module as get_ops

import turibolt as bolt


class BoltCallback(TrainingProcessCallback):
    """
    Callback for reporting metrics to bolt, for internal use.

    TODO: Remove for public export.
    """

    def __init__(self):
        pass

    @property
    def bolt(self):
        # Not installed by default.
        import turibolt as bolt
        return bolt

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Submits metrics of this central iteration to bolt job.
        """
        if get_ops().distributed.global_rank == 0:
            # Wandb package already uses a multithreaded solution
            # to submit log requests to server, such that this
            # call will not be blocking until server responds.
            metrics_dict = aggregate_metrics.to_simple_dict()
            if len(metrics_dict) > 0:
                if self.bolt.get_task(self.bolt.get_current_task_id()).parent_id is None:
                    self.bolt.send_metrics(
                        aggregate_metrics.to_simple_dict(),
                        iteration=central_iteration)
                else:
                    self.bolt.send_metrics(
                        aggregate_metrics.to_simple_dict(),
                        iteration=central_iteration,
                        report_as_parent=True)

        return False, Metrics()

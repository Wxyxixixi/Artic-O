from .base_trainer import BaseTrainer
from .recon_trainer import ReconTrainer
from .artic_o_trainer import ArticOTrainer

# Maps ``cfg.trainer_class`` to the class eval.py instantiates.
# ArticOTrainer is the one the released config uses; the other two are its
# base classes and are exported so subclassing stays discoverable.
TRAINER_REGISTRY = {
    "BaseTrainer": BaseTrainer,
    "ReconTrainer": ReconTrainer,
    "ArticOTrainer": ArticOTrainer,
}

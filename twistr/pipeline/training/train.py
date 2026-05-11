from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from twistr.pipeline.config import MLConfig, load_ml_config
from twistr.pipeline.datasets.datamodule import ExamplesDataModule
from twistr.pipeline.models.lightning_module import ExamplesModule

# A100 / H100 / A6000 all support TF32 matmul. Lightning's default
# precision is float32 with strict (slow) matmul; switching to "high"
# unlocks the tensor-core fast path with a tiny accuracy hit. The
# Lightning warning at run start explicitly recommends this.
torch.set_float32_matmul_precision("high")

# Surface the calibrator's cache-hit / per-N max_B messages to stdout so it's
# obvious whether training reused a cached calibration or re-probed.
logging.getLogger("twistr.pipeline.training.batch_calibration").addHandler(
    logging.StreamHandler()
)
logging.getLogger("twistr.pipeline.training.batch_calibration").setLevel(logging.INFO)


def trainer_kwargs(cfg: MLConfig) -> dict:
    if cfg.num_gpus == 0:
        return {"accelerator": "cpu", "devices": 1}
    if cfg.num_gpus == 1:
        return {"accelerator": "gpu", "devices": 1}
    # `use_distributed_sampler=False`: Lightning can't auto-wrap our custom
    # `batch_sampler=` (LengthBucketBatchSampler) into a DistributedSampler.
    # The sampler reads `trainer.world_size` / `trainer.global_rank` itself
    # and emits a balanced bucket count per rank, so we own the partition.
    return {
        "accelerator": "gpu",
        "devices": cfg.num_gpus,
        "strategy": "ddp",
        "use_distributed_sampler": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("runtime/configs/ml.yaml"))
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume from (must match current architecture).",
    )
    args = parser.parse_args()

    cfg = load_ml_config(args.config)
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = ExamplesDataModule(cfg=cfg)
    model = ExamplesModule(cfg=cfg)
    logger = WandbLogger(project="helix_design", config=cfg.__dict__, log_model="all")
    # Per-launch subdirectory keyed by timestamp. Each launch's checkpoints
    # are isolated from prior runs whose architecture or hyperparams may
    # differ, so a stale last.ckpt can't trip up a fresh launch via Lightning's
    # strict state-dict load. Crash recovery is opt-in via --resume.
    ckpt_dir = Path(cfg.checkpoint_dir) / f"run_{int(time.time())}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{step}-{val/loss_total:.4f}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=20,
        save_last=True,
        auto_insert_metric_name=False,
    )
    resume_path = str(args.resume) if args.resume is not None else None
    if resume_path is not None:
        logging.getLogger(__name__).info("Resuming from %s", resume_path)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        # Skip Lightning's pre-training val pass. With dynamic batching the
        # first val batches use the largest B from calibration, which on a
        # network-filesystem dataset means a single worker loads hundreds of
        # files sequentially before training can start.
        num_sanity_val_steps=0,
        # Every 10 steps: dense enough to see early dynamics, sparse enough
        # to keep wandb history manageable for 50k-step runs.
        log_every_n_steps=10,
        **trainer_kwargs(cfg),
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()

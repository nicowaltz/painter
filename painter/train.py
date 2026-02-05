import os, glob
import csv
import time
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from config import TrainerConfig, PainterConfig
from model import Painter
from dataloader import InfiniteDataLoader, create_dataloader

class Trainer:
    def __init__(self, model_config: PainterConfig, train_config: TrainerConfig):
        self.model_config = model_config
        self.train_config = train_config

        self.ddp = torch.distributed.is_available() and torch.distributed.is_initialized() 

        if self.ddp: self._setup_ddp_training()
        else: self._setup_training()

        self._setup_dataloaders()
        self._setup_epoch_tracking()
        self._setup_logging()

    def _log(self, message: str):
        if not self.is_master: return 

        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        tqdm.write(message)

    def _setup_logging(self):
        os.makedirs(self.train_config.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.train_config.log_dir, "train_log.log")
        self.loss_log_file = os.path.join(self.train_config.log_dir, "loss.csv")

        if self.is_master and not os.path.exists(self.loss_log_file):
            with open(self.loss_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "loss", "lr", "epoch", "tokens_per_sec"])

    def _log_loss(self, step: int, loss: float, lr: float, epoch: int, tokens_per_sec: float):
        if not self.is_master:
            return
        with open(self.loss_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{loss:.6f}", f"{lr:.2e}", epoch, f"{tokens_per_sec:.0f}"])

    def _setup_epoch_tracking(self):

        train_dir = os.path.join(self.train_config.data_dir, "train")

        metadata_path = os.path.join(train_dir, "metadata.npy")
        total_tokens = 0
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            total_tokens = int(metadata.get("total_rows", 0))
        else:
            shard_files = sorted(glob.glob(os.path.join(train_dir, "*.npy")))
            shard_files = [f for f in shard_files if "metadata" not in f]
            if shard_files:
                # cheap estimate: assume all shards similar length
                first = np.load(shard_files[0], mmap_mode="r")
                total_tokens = int(len(first) * len(shard_files))

        micro_batch_size = self.train_config.batch_size          # per-rank batch size
        context_len = self.model_config.context_len
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps

        tokens_per_optimizer_step = micro_batch_size * context_len * gradient_accumulation_steps

        self.steps_per_epoch = max(1, total_tokens // max(1, tokens_per_optimizer_step))
        self.step = 0
        self.epoch = 0

        self._log(
            f"Steps per epoch: {self.steps_per_epoch} | "
            f"total_tokens≈{total_tokens:,} | "
            f"tokens/opt_step={tokens_per_optimizer_step:,} | "
        )

    def _create_optimizer(self, model):
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _create_scheduler(self, optimizer):
        """Setup scheduler with warmup depending on config"""
        if self.train_config.scheduler is None or self.train_config.scheduler == "none":
            main_scheduler = None
        elif self.train_config.scheduler == "cosine":
            T_max = self.train_config.epochs - self.train_config.warmup_epochs
            main_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=self.train_config.min_lr)
        else:
            raise ValueError(f"Scheduler type '{self.train_config.scheduler}' not supported")

        if self.train_config.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=self.train_config.warmup_factor, total_iters=self.train_config.warmup_epochs
            )

            if main_scheduler is None:
                return warmup_scheduler
            else:
                return SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.train_config.warmup_epochs],
                )

        return main_scheduler

    def _reduce_tensor(self, tensor: torch.Tensor):
        if not self.ddp:
            return tensor
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
        return tensor

    def _get_batch(self, loader: InfiniteDataLoader):
        """Get a batch from the dataloader."""
        x, targets = next(loader)
        x = x.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return x, targets

    def _train_step(self):
        self.optimizer.zero_grad()
        accumulated_loss = 0.0

        for acc_step in range(self.gradient_accumulation_steps):
            x, targets = self._get_batch(self.train_loader) # batch_size, context_len, 2

            # Karpathy hack
            if self.ddp:
                self.model.require_backward_grad_sync = (acc_step == self.gradient_accumulation_steps - 1)

            logits, loss = self.model(x, targets)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.detach()

        if self.train_config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip)

        self.optimizer.step()

        self.step += 1

        return self._reduce_tensor(accumulated_loss).item()

    def _train_epoch(self):
        tokens_per_step = (
            self.train_config.batch_size
            * self.model_config.context_len
            * self.gradient_accumulation_steps
        )

        t0 = time.time()
        epoch_loss = 0.0
        for _ in tqdm(range(self.steps_per_epoch), desc=f"Epoch {self.epoch}", leave=False, disable=not self.is_master):
            loss = self._train_step()
            epoch_loss += loss

        dt = time.time() - t0
        avg_loss = epoch_loss / self.steps_per_epoch
        tokens_per_sec = tokens_per_step * self.steps_per_epoch / dt if dt > 0 else 0
        lr = self.optimizer.param_groups[0]["lr"]

        self._log(
            f"Epoch {self.epoch} | loss={avg_loss:.4f} | lr={lr:.2e} | "
            f"tok/s={tokens_per_sec:.0f}"
        )
        self._log_loss(self.step, avg_loss, lr, self.epoch, tokens_per_sec)

        self.epoch += 1
        self.train_loader.set_epoch(self.epoch)

        if self.scheduler is not None:
            self.scheduler.step()

        return avg_loss

    def _save_checkpoint(self, is_best: bool = False):
        if not self.is_master:
            return

        model_to_save = self.model.module if self.ddp else self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod

        checkpoint = {
            "step": self.step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "model_config": self.model_config,
            "train_config": self.train_config,
            "best_val_loss": self.best_val_loss,
            "epoch": self.epoch
        }

        path = os.path.join(self.train_config.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, path)

        if is_best:
            path = os.path.join(self.train_config.checkpoint_dir, "best.pt")
            torch.save(checkpoint, path)

        if self.step % self.train_config.save_interval == 0:
            path = os.path.join(
                self.train_config.checkpoint_dir,
                f"step_{self.step:06d}.pt"
            )
            torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self._log(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        model_to_load = self.model.module if self.ddp else self.model
        if hasattr(model_to_load, "_orig_mod"):
            model_to_load = model_to_load._orig_mod

        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    @torch.no_grad()
    def _evaluate(self, loader: InfiniteDataLoader):
        """Evaluate on validation set with and without document position."""
        self.model.eval()
        total_nodrop_loss = torch.tensor(0.0, device=self.device)
        total_drop_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.train_config.eval_steps):
            x, targets = self._get_batch(loader)
            x_drop = x.clone()
            x_drop[:, :, 1] = 0
            logits_nodrop, _ = self.model(x)
            logits_drop, _ = self.model(x_drop)
            loss_nodrop = F.cross_entropy(logits_nodrop.view(-1, logits_nodrop.size(-1)), targets.view(-1))
            loss_drop = F.cross_entropy(logits_drop.view(-1, logits_drop.size(-1)), targets.view(-1))
            total_nodrop_loss += loss_nodrop
            total_drop_loss += loss_drop

        avg_nodrop_loss = self._reduce_tensor(total_nodrop_loss / self.train_config.eval_steps)
        avg_drop_loss = self._reduce_tensor(total_drop_loss / self.train_config.eval_steps)

        self.model.train()
        return avg_nodrop_loss.item(), avg_drop_loss.item()

    def _setup_ddp_training(self):
        self.rank = torch.distributed.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
        self.world_size = torch.distributed.get_world_size()
        self.is_master = self.rank == 0 
        self.device = torch.device(f"cuda:{int(self.local_rank % torch.cuda.device_count())}")
        self.gradient_accumulation_steps = self.train_config.gradient_accumulation_steps // self.world_size

        self.model = Painter(self.model_config)
        self.model = self.model.to(self.device)
        if self.train_config.compile_model: self.model = torch.compile(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank], output_device=self.local_rank,
            find_unused_parameters=False
        )

        self.optimizer = self._create_optimizer(self.model)
        self.scheduler = self._create_scheduler(self.optimizer)

        self.best_val_loss = float("inf")

    def _setup_training(self):
        self.rank = 0
        self.local_rank = 0 
        self.world_size = 1 
        self.is_master = True 
        self.device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "mps" if hasattr(torch, 'mps') and torch.mps.is_available() else "cpu"
        self.gradient_accumulation_steps = self.train_config.gradient_accumulation_steps


        self.model = Painter(self.model_config)
        self.model = self.model.to(self.device)
        if self.train_config.compile_model: self.model = torch.compile(self.model)

        self.optimizer = self._create_optimizer(self.model)
        self.scheduler = self._create_scheduler(self.optimizer)

        self.best_val_loss = float("inf")

    def _setup_dataloaders(self):
        self.train_loader = create_dataloader(
            self.train_config.data_dir,
            self.model_config.context_len,
            self.train_config.batch_size,
            num_workers=self.train_config.num_dataloader_workers,
        )
        self.train_loader.set_epoch(0)

        self.val_loader = create_dataloader(
            self.train_config.data_dir,
            self.model_config.context_len,
            self.train_config.batch_size,
            num_workers=1,
            validation=True,
        )
        # Deterministic due to set seed
        self.val_loader.set_epoch(0)


    def train_loop(self):
        self.model.train()
        epochs_without_improvement = 0

        try:
            for _ in range(self.train_config.epochs):
                loss = self._train_epoch()


                if self.ddp: torch.distributed.barrier()

                val_loss_nodrop, val_loss_drop = self._evaluate(self.val_loader)
                val_loss = (val_loss_nodrop + val_loss_drop) / 2

                self._log(
                    f"Epoch {self.epoch} | Step {self.step} | train_loss={loss:.4f} | "
                    f"val_loss_nodrop={val_loss_nodrop:.4f} | val_loss_drop={val_loss_drop:.4f}| "
                    f"val_loss={val_loss:.4f}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(is_best=True)
                else: 
                    epochs_without_improvement += 1

                self._save_checkpoint()

                if self.ddp: torch.distributed.barrier()
        finally:
            if self.ddp: torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train Painter model")

    # Model config
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-embed", type=int, default=768)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-pred-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Trainer config
    parser.add_argument("--data-dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--warmup-factor", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", action="store_false", dest="compile")
    parser.add_argument("--num-dataloader-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--ddp", type=str, default=None, help="Run in ddp mode")

    args = parser.parse_args()

    model_config = PainterConfig(
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        n_embed=args.n_embed,
        context_len=args.context_len,
        n_heads=args.n_heads,
        n_pred_heads=args.n_pred_heads,
        dropout=args.dropout,
    )

    train_config = TrainerConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        log_dir=args.log_dir,
        eval_steps=args.eval_steps,
        save_interval=args.save_interval,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        checkpoint_dir=args.checkpoint_dir,
        compile_model=args.compile,
        num_dataloader_workers=args.num_dataloader_workers,
    )

    

    
    if args.ddp == "yes": 
        torch.distributed.init_process_group(backend="nccl")
    trainer = Trainer(model_config, train_config)

    if args.resume:
        trainer._load_checkpoint(args.resume)

    trainer.train_loop()

if __name__ == "__main__":
    main()

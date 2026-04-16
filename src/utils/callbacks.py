import torch
import torch.distributed as dist
from pytorch_lightning import Callback
from utils.scheduler import CurriculumScheduler

class SchedulerUpdateCallback(Callback):
    def __init__(self, config, scheduler: CurriculumScheduler):
        super().__init__()
        self.config = config
        self.scheduler = scheduler

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Update scheduler with trainer's global_step at the start of each training batch.
        current_step = trainer.global_step
        self.scheduler.update(current_step)

        # Log current probabilities periodically to verify they change correctly.
        if current_step % self.config.trainer.log_steps == 0:
            probs = self.scheduler.get_probabilities()
            pl_module.log_dict({
                'sampler/prob_task1': probs[0],
                'sampler/prob_task2': probs[1],
                'sampler/prob_task3': probs[2],
            }, on_step=True, logger=True)

class TaskLossLogger(Callback):
    def __init__(self, log_every_n_steps, num_tasks=3):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_tasks = num_tasks
        self.task_losses_buffer = {i: [] for i in range(1, num_tasks + 1)}
        self.task_grad_norms_buffer = {i: [] for i in range(1, num_tasks + 1)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after every batch."""
        # Store loss.
        task = pl_module.current_task
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        self.task_losses_buffer[task].append(loss.detach().item())

        # Log at every log_every_n_steps interval.
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            self._log_metrics(trainer, pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Store gradient norm."""
        if not hasattr(pl_module, 'current_task'):
            return

        task = pl_module.current_task

        # Compute grad norm.
        if pl_module.config.trainer.strategy == 'fsdp':
            grad_norm = trainer.model.clip_grad_norm_(pl_module.config.trainer.gradient_clip_val)
        elif pl_module.config.trainer.strategy == 'deepspeed':
            grad_norm = pl_module.get_deepspeed_grad_norm()
        else:
            grad_norm = pl_module.compute_total_grad_norm()

        if grad_norm is not None and grad_norm != float('inf'):
            grad_norm_value = grad_norm if isinstance(grad_norm, float) else grad_norm.item()
            self.task_grad_norms_buffer[task].append(grad_norm_value)

    def _log_metrics(self, trainer, pl_module):
        """Collect buffers from all GPUs, compute mean, and log."""

        if not (dist.is_available() and dist.is_initialized()):
            # Single-GPU case.
            log_dict = self._compute_local_metrics(trainer, pl_module)
            self._write_logs(trainer, log_dict)
            self._reset_buffers()
            return

        # Multi-GPU: collect via all_gather.
        world_size = dist.get_world_size()
        device = pl_module.device

        # Compute local sums and counts.
        local_loss_sum = torch.zeros(self.num_tasks, device=device, dtype=torch.float)
        local_loss_count = torch.zeros(self.num_tasks, device=device, dtype=torch.long)
        local_grad_sum = torch.zeros(self.num_tasks, device=device, dtype=torch.float)
        local_grad_count = torch.zeros(self.num_tasks, device=device, dtype=torch.long)

        for i, t in enumerate(range(1, self.num_tasks + 1)):
            if self.task_losses_buffer[t]:
                local_loss_sum[i] = sum(self.task_losses_buffer[t])
                local_loss_count[i] = len(self.task_losses_buffer[t])

            if self.task_grad_norms_buffer[t]:
                local_grad_sum[i] = sum(self.task_grad_norms_buffer[t])
                local_grad_count[i] = len(self.task_grad_norms_buffer[t])

        # All-gather
        loss_sum_list = [torch.zeros_like(local_loss_sum) for _ in range(world_size)]
        loss_count_list = [torch.zeros_like(local_loss_count) for _ in range(world_size)]
        grad_sum_list = [torch.zeros_like(local_grad_sum) for _ in range(world_size)]
        grad_count_list = [torch.zeros_like(local_grad_count) for _ in range(world_size)]

        dist.all_gather(loss_sum_list, local_loss_sum)
        dist.all_gather(loss_count_list, local_loss_count)
        dist.all_gather(grad_sum_list, local_grad_sum)
        dist.all_gather(grad_count_list, local_grad_count)

        # Compute global averages.
        global_loss_sum = torch.stack(loss_sum_list).sum(dim=0)
        global_loss_count = torch.stack(loss_count_list).sum(dim=0)
        global_grad_sum = torch.stack(grad_sum_list).sum(dim=0)
        global_grad_count = torch.stack(grad_count_list).sum(dim=0)

        # Build log dict.
        log_dict = {
            'train/lr': trainer.optimizers[0].param_groups[0]['lr'],
            'train/global_step': float(trainer.global_step),
        }

        for i, t in enumerate(range(1, self.num_tasks + 1)):
            # Loss
            if global_loss_count[i] > 0:
                log_dict[f'train/task_{t}/loss'] = (global_loss_sum[i] / global_loss_count[i]).item()
                log_dict[f'train/task_{t}/loss_count'] = global_loss_count[i].item()

            # Grad norm
            if global_grad_count[i] > 0:
                log_dict[f'train/task_{t}/grad_norm'] = (global_grad_sum[i] / global_grad_count[i]).item()
                log_dict[f'train/task_{t}/grad_count'] = global_grad_count[i].item()

        # Log only from rank 0.
        if trainer.is_global_zero:
            self._write_logs(trainer, log_dict)

        # Reset buffers.
        self._reset_buffers()

    def _compute_local_metrics(self, trainer, pl_module):
        """Compute metrics for single-GPU mode."""
        log_dict = {
            'train/lr': trainer.optimizers[0].param_groups[0]['lr'],
            'train/global_step': float(trainer.global_step),
        }

        for t in range(1, self.num_tasks + 1):
            if self.task_losses_buffer[t]:
                log_dict[f'train/task_{t}/loss'] = sum(self.task_losses_buffer[t]) / len(self.task_losses_buffer[t])
                log_dict[f'train/task_{t}/loss_count'] = len(self.task_losses_buffer[t])

            if self.task_grad_norms_buffer[t]:
                log_dict[f'train/task_{t}/grad_norm'] = sum(self.task_grad_norms_buffer[t]) / len(self.task_grad_norms_buffer[t])
                log_dict[f'train/task_{t}/grad_count'] = len(self.task_grad_norms_buffer[t])

        return log_dict

    def _write_logs(self, trainer, log_dict):
        """Write metrics directly to the logger."""
        if trainer.logger is not None:
            # Log directly to WandB, TensorBoard, etc.
            trainer.logger.log_metrics(log_dict, step=trainer.global_step)

        # Update progress bar (optional).
        if hasattr(trainer, 'progress_bar_callback') and trainer.progress_bar_callback:
            trainer.progress_bar_callback.main_progress_bar.set_postfix(log_dict)

    def _reset_buffers(self):
        """Reset buffers."""
        self.task_losses_buffer = {i: [] for i in range(1, self.num_tasks + 1)}
        self.task_grad_norms_buffer = {i: [] for i in range(1, self.num_tasks + 1)}

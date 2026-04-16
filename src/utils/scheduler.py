# Ref: https://gaussian37.github.io/dl-pytorch-lr_scheduler/#cosineannealingwarmrestarts-1

from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineDecayWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, warmup_iter, max_iter, eta_min=0, eta_max=1.5e-4, last_epoch=-1):
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.iteration = 0

        super(CosineDecayWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.warmup_iter:
            lr = self.eta_max * self.iteration / self.warmup_iter
        elif self.iteration > self.max_iter:
            lr = self.eta_min
        else:
            decay_ratio = (self.iteration - self.warmup_iter) / (self.max_iter - self.warmup_iter)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.eta_min + (self.eta_max - self.eta_min) * coeff
        return lr

    def step(self, epoch=None):
        self.iteration += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = param_group["lr_scale"] * lr
            else:
                param_group['lr'] = lr

import numpy as np

class CurriculumScheduler:
    def __init__(self, init_probs: list, total_steps: int, task_base_weights: list, overlap_factor: float = 1.5, min_weight_ratio: float = 0.1):
        self.total_steps = total_steps
        self.task_base_weights = task_base_weights
        self.init_probs = init_probs
        self.overlap_factor = overlap_factor
        self.min_weight_ratio = min_weight_ratio

        assert len(init_probs) == 3, "init_probs must have 3 elements."

        self._current_step = 0
        self.current_probs = np.array(init_probs)

        print("🚀 CurriculumScheduler initialized!")
        print(f"Total steps: {self.total_steps}, Base weights: {self.task_base_weights}")

    def _calculate_weights(self, current_step: int) -> dict:
        progress = current_step / (self.total_steps - 1) if self.total_steps > 1 else 0

        activation1 = np.cos(progress * 0.5 * np.pi)
        activation2 = np.sin(progress * np.pi)
        activation3 = np.sin(progress * 0.5 * np.pi)

        activations = np.array([activation1, activation2, activation3]) ** self.overlap_factor

        total_activation = np.sum(activations)
        if total_activation < 1e-9:
            activations = np.array([1/3, 1/3, 1/3])
        else:
            activations /= total_activation

        total_min_weight = 3 * self.min_weight_ratio
        timing_based_weights = (activations * (1.0 - total_min_weight)) + self.min_weight_ratio

        base_weights_array = np.array([
            self.task_base_weights[0],
            self.task_base_weights[1],
            self.task_base_weights[2]
        ])
        corrected_weights = timing_based_weights * base_weights_array

        # Renormalize so probabilities sum to 1.
        final_probabilities = corrected_weights / np.sum(corrected_weights)

        return final_probabilities

    def update(self, current_step: int):
        self._current_step = current_step
        self.current_probs = self._calculate_weights(current_step)

    def get_probabilities(self) -> np.ndarray:
        return self.current_probs

    def state_dict(self):
        """Return scheduler state as a dict for checkpoint saving."""
        return {
            '_current_step': self._current_step,
            'current_probs': self.current_probs,
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler state from a dict."""
        self._current_step = state_dict['_current_step']
        self.current_probs = state_dict['current_probs']
        print(f"✅ CurriculumScheduler state has been restored to step {self._current_step}.")

class CosineCurriculumScheduler:
    def __init__(self, init_probs: list, total_steps: int, min_prob: float = 0.1):
        """
        Curriculum scheduler with cosine decay.
        - Task 1: fixed probability
        - Task 2: cosine decay from init_probs[1] to min_prob
        - Task 3: increases to absorb remaining probability
        """
        self.total_steps = total_steps
        self.init_probs = np.array(init_probs)
        self.min_prob = min_prob

        assert len(init_probs) == 3, "init_probs must have 3 elements."
        assert np.isclose(np.sum(init_probs), 1.0), "init_probs must sum to 1."

        self._current_step = 0
        self.current_probs = self.init_probs.copy()

        print("🚀 CosineCurriculumScheduler initialized!")
        print(f"Total steps: {self.total_steps}, Initial probabilities: {self.init_probs}, Min probability: {self.min_prob}")

    def _calculate_weights(self, current_step: int) -> np.ndarray:
        # Compute training progress in [0.0, 1.0].
        progress = current_step / (self.total_steps - 1) if self.total_steps > 1 else 1.0

        # Task 1: keep fixed.
        prob1 = self.init_probs[0]

        # Task 2: cosine decay.
        start_prob2 = self.init_probs[1]
        end_prob2 = self.min_prob

        # 1. Cosine factor: 1 at progress=0, 0 at progress=1.
        cosine_factor = 0.5 * (1 + np.cos(progress * np.pi))

        # 2. Interpolate between start_prob2 and end_prob2.
        prob2 = end_prob2 + (start_prob2 - end_prob2) * cosine_factor

        # Task 3: absorb remaining probability.
        prob3 = 1.0 - prob1 - prob2

        final_probabilities = np.array([prob1, prob2, prob3])

        return final_probabilities / np.sum(final_probabilities)

    def update(self, current_step: int):
        self._current_step = current_step
        self.current_probs = self._calculate_weights(current_step)

    def get_probabilities(self) -> np.ndarray:
        return self.current_probs

    def state_dict(self):
        """Return scheduler state as a dict for checkpoint saving."""
        return {
            '_current_step': self._current_step,
            'current_probs': self.current_probs,
        }

    def load_state_dict(self, state_dict):
        """Restore scheduler state from a dict."""
        self._current_step = state_dict['_current_step']
        self.current_probs = state_dict['current_probs']
        print(f"✅ CosineCurriculumScheduler state has been restored to step {self._current_step}.")

class ConstantCurriculumScheduler:
    def __init__(self, init_probs: list):
        """
        Curriculum scheduler with constant probabilities.
        All task probabilities are fixed at init_probs throughout training.
        """
        self.init_probs = np.array(init_probs)

        # Validate init_probs.
        assert len(init_probs) == 3, "init_probs must have 3 elements."
        assert np.isclose(np.sum(init_probs), 1.0), "init_probs must sum to 1."

        self.current_probs = self.init_probs.copy()

        print("🚀 ConstantCurriculumScheduler initialized!")
        print(f"Constant probabilities: {self.current_probs}")

    def update(self, current_step: int):
        """
        No-op: this scheduler does not update probabilities.
        Kept for interface compatibility.
        """
        pass

    def get_probabilities(self) -> np.ndarray:
        """Return the fixed probabilities."""
        return self.current_probs

    def state_dict(self):
        """
        No state to save as probabilities never change.
        Returns an empty dict for interface compatibility.
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        No state to restore as probabilities never change.
        No-op for interface compatibility.
        """
        print("✅ ConstantCurriculumScheduler has no state to restore.")
        pass

class StepSwitchCurriculumScheduler:
    def __init__(self, init_probs: list, target_probs: list, switch_step: int):
        """
        Curriculum scheduler that switches probability distribution at a given step.
        - Before switch_step: use init_probs
        - After switch_step: use target_probs
        """
        self.init_probs = np.array(init_probs)
        self.target_probs = np.array(target_probs)
        self.switch_step = switch_step

        # Validate inputs.
        assert len(init_probs) == 3, "init_probs must have 3 elements."
        assert len(target_probs) == 3, "target_probs must have 3 elements."
        assert np.isclose(np.sum(init_probs), 1.0), "init_probs must sum to 1."
        assert np.isclose(np.sum(target_probs), 1.0), "target_probs must sum to 1."

        # Set initial state.
        self.current_probs = self.init_probs.copy()
        self.has_switched = False

        print("🚀 StepSwitchCurriculumScheduler initialized!")
        print(f"Phase 1 (Step < {self.switch_step}): {self.init_probs}")
        print(f"Phase 2 (Step >= {self.switch_step}): {self.target_probs}")

    def update(self, current_step: int):
        """Update probability distribution based on current step."""
        if current_step >= self.switch_step:
            # After switch point.
            self.current_probs = self.target_probs

            # Log only once at the moment of switching.
            if not self.has_switched:
                print(f"\n[Curriculum] Step {current_step}: Switching probabilities to {self.target_probs}")
                self.has_switched = True
        else:
            # Before switch point (or rewound via checkpoint load).
            self.current_probs = self.init_probs
            self.has_switched = False

    def get_probabilities(self) -> np.ndarray:
        return self.current_probs

    def state_dict(self):
        # Step info is managed by the Trainer; no internal state to save.
        # (has_switched can be stored if needed, but update() auto-recovers it.)
        return {}

    def load_state_dict(self, state_dict):
        pass

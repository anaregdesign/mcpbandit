from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


@dataclass
class ArmState:
    """Sufficient statistics for a single linear contextual bandit arm."""

    L: NDArray[np.float64]
    b: NDArray[np.float64]
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @classmethod
    def initial(cls, context_length: int, lam: float = 1.0) -> "ArmState":
        """Initialize arm statistics with Tikhonov regularization.

        Args:
            context_length: Dimensionality of the context vector.
            lam: Regularization strength (lambda). Think of this as a safety
                padding added before any data is seen: larger values make the
                model start more cautiously so early observations do not swing
                the estimates too hard, while smaller values let the model
                adapt faster at the cost of being noisier at the beginning.

        Returns:
            A new `ArmState` with initialized design matrix factor and response vector.
        """
        L = np.sqrt(lam) * np.identity(context_length, dtype=np.float64)
        b = np.zeros((context_length, 1), dtype=np.float64)
        return cls(L=L, b=b)

    def update(self, reward: float, context: NDArray[np.float64]) -> None:
        """Thread-safe incorporation of a new (context, reward) observation."""
        context_vec = context.reshape(-1, 1)
        with self._lock:
            self._cholesky_rank_one_update(context_vec)
            self.b += reward * context_vec

    def _cholesky_rank_one_update(self, x: NDArray[np.float64]) -> None:
        """Apply an in-place rank-one update to the Cholesky factor."""
        v = x.flatten().astype(np.float64, copy=True)
        for i in range(self.L.shape[0]):
            r = np.hypot(self.L[i, i], v[i])
            c = r / self.L[i, i]
            s = v[i] / self.L[i, i]
            self.L[i, i] = r
            if i + 1 < self.L.shape[0]:
                self.L[i + 1 :, i] = (self.L[i + 1 :, i] + s * v[i + 1 :]) / c
                v[i + 1 :] = c * v[i + 1 :] - s * self.L[i + 1 :, i]


@dataclass
class Arm(Generic[T]):
    id: int
    body: T
    state: ArmState


@dataclass
class BanditRegistry(ABC, Generic[T]):
    context_length: int
    arms: list[Arm[T]] = field(default_factory=list)

    def add(self, body: T) -> None:
        """Add a new arm to the policy with initialized statistics."""
        arm = Arm(
            id=len(self.arms),
            body=body,
            state=ArmState.initial(self.context_length),
        )
        self.arms.append(arm)

    def observe(self, arm_id: int, reward: float, context: NDArray[np.float64]) -> None:
        """Update internal statistics after observing reward."""
        arm = self.arms[arm_id]
        arm.state.update(reward, context)

    @abstractmethod
    def select(self, context: NDArray[np.float64]) -> Arm[T]:
        """Choose an arm based on the provided context."""
        pass


@dataclass
class ThompsonSamplingRegistry(BanditRegistry[T], Generic[T]):
    """Linear Thompson Sampling policy using a Gaussian posterior.

    Args:
        arms: Bandit arm states containing their running estimates.
        alpha: Exploration scale. Works like a temperature knob for the random
            sampling step: higher values inject more randomness so the policy
            keeps trying less-certain arms, and lower values make it act more
            greedily based on the current estimates. The default of 0.3 is a
            conservative choice that balances exploration and exploitation in
            many practical scenarios.
    """

    alpha: float = 0.3

    def select(self, context: NDArray[np.float64]) -> Arm[T]:
        sampled_means: list[float] = []
        context_vec = context.reshape(-1, 1)
        for arm in self.arms:
            with arm.state._lock:
                y = np.linalg.solve(arm.state.L, arm.state.b)
                mu_hat = np.linalg.solve(arm.state.L.T, y)

                z = np.random.normal(size=mu_hat.shape)
                perturbation = self.alpha * np.linalg.solve(arm.state.L.T, z)
                sampled_theta = mu_hat + perturbation

                sampled_mean = float((context_vec.T @ sampled_theta).item())
                sampled_means.append(sampled_mean)

        chosen_index = int(np.argmax(sampled_means))
        return self.arms[chosen_index]


@dataclass
class UCBRegistry(BanditRegistry[T], Generic[T]):
    """Linear UCB policy with ellipsoidal confidence bounds.

    Args:
        arms: Bandit arm states containing their running estimates.
        alpha: Exploration weight for the confidence bonus. Higher values mean
            the policy adds a larger safety margin for uncertainty, encouraging
            more exploration; smaller values favor sticking with the best-known
            arm sooner. The default of 0.5 is a typical setting that encourages
            learning without being overly cautious.
    """

    alpha: float = 0.5

    def select(self, context: NDArray[np.float64]) -> Arm[T]:
        """Compute upper confidence bounds and pick the arm with the highest score.

        Args:
            context: Context vector for the decision point.

        Returns:
            The selected arm.
        """
        ucb_values: list[float] = []
        context_vec = context.reshape(-1, 1)
        for arm in self.arms:
            with arm.state._lock:
                y = np.linalg.solve(arm.state.L, arm.state.b)
                mu_hat = np.linalg.solve(arm.state.L.T, y)

                y = np.linalg.solve(arm.state.L, context_vec)  # y = L^{-1} x
                uncertainty = np.sqrt(float((y.T @ y).item()))

                ucb_value = (
                    float((context_vec.T @ mu_hat).item()) + self.alpha * uncertainty
                )
                ucb_values.append(ucb_value)
        chosen_index = int(np.argmax(ucb_values))
        return self.arms[chosen_index]

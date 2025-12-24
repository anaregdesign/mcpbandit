from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from mcpbandit.bandit import (
    ArmState,
    ThompsonSamplingRegistory,
    UCBRegistory,
)


def _simulate(registory_factory: callable, *, rounds: int = 400) -> np.ndarray:
    """Run a registory against two arms with binary 5-D contexts."""
    np.random.seed(0)
    rng = np.random.default_rng(1)

    context_dim = 5
    registory = registory_factory(context_dim)
    for _ in range(2):
        registory.add(body=None)

    true_thetas = [
        np.array([0.1, 0.1, 0.05, 0.05, 0.05], dtype=float).reshape(-1, 1),
        np.array([0.6, 0.6, 0.2, 0.2, 0.2], dtype=float).reshape(-1, 1),
    ]

    pulls = np.zeros(len(registory.arms), dtype=int)
    for _ in range(rounds):
        context = rng.integers(0, 2, size=(context_dim,)).astype(float)
        chosen_arm = registory.select(context)
        reward = float(context @ true_thetas[chosen_arm.id].flatten())
        registory.observe(chosen_arm.id, reward, context)
        pulls[chosen_arm.id] += 1
    return pulls


@pytest.mark.parametrize(
    ("name", "registory_factory"),
    [
        (
            "thompson",
            lambda context_dim: ThompsonSamplingRegistory(
                context_length=context_dim, alpha=0.3
            ),
        ),
        (
            "ucb",
            lambda context_dim: UCBRegistory(
                context_length=context_dim, alpha=0.5
            ),
        ),
    ],
)
def test_high_reward_arm_is_selected_more_often(
    name: str, registory_factory: callable
) -> None:
    pulls = _simulate(registory_factory)
    assert pulls[1] > pulls[0], (
        f"[{name}] high-reward arm should be pulled more often: {pulls}"
    )
    assert pulls[1] / pulls.sum() >= 0.7, (
        f"[{name}] high-reward arm pull ratio too low: {pulls}"
    )


def test_linear_bandit_arm_state_update_is_thread_safe() -> None:
    dim = 4
    rng = np.random.default_rng(123)
    contexts = rng.standard_normal((20, dim))
    rewards = rng.standard_normal(20)

    threaded = ArmState.initial(dim)
    applied_order: list[int] = []

    def worker(idx: int) -> None:
        threaded.update(float(rewards[idx]), contexts[idx])
        applied_order.append(idx)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(len(contexts))))

    expected = ArmState.initial(dim)
    for idx in applied_order:
        expected.update(float(rewards[idx]), contexts[idx])

    assert np.allclose(threaded.L, expected.L)
    assert np.allclose(threaded.b, expected.b)

import asyncio

import pytest

from mcpbandit.context import async_lru_cache


@pytest.mark.asyncio
async def test_cache_hits_reuse_result() -> None:
    call_count = 0

    @async_lru_cache(maxsize=4)
    async def add(a: int, b: int = 0) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        return a + b

    first = await add(1, b=2)
    second = await add(a=1, b=2)  # kwarg order should not matter

    assert first == 3
    assert second == 3
    assert call_count == 1


@pytest.mark.asyncio
async def test_lru_eviction_recomputes_old_entries() -> None:
    call_count = 0

    @async_lru_cache(maxsize=2)
    async def echo(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        return x

    await echo(1)
    await echo(2)
    await echo(3)  # evicts key 1 (LRU)
    await echo(2)  # still cached
    await echo(1)  # should recompute

    assert call_count == 4


@pytest.mark.asyncio
async def test_failed_calls_do_not_poison_cache() -> None:
    attempts = 0

    @async_lru_cache(maxsize=2)
    async def sometimes_fails(x: int) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return x

    with pytest.raises(RuntimeError):
        await sometimes_fails(5)

    result = await sometimes_fails(5)

    assert result == 5
    assert attempts == 2


@pytest.mark.asyncio
async def test_concurrent_calls_share_same_task() -> None:
    call_count = 0
    gate = asyncio.Event()

    @async_lru_cache(maxsize=2)
    async def slow(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await gate.wait()
        return x

    # Start two concurrent calls with identical args; they should await the same task.
    t1 = asyncio.create_task(slow(9))
    t2 = asyncio.create_task(slow(9))

    await asyncio.sleep(0.01)  # allow both tasks to reach the cache
    assert call_count == 1

    gate.set()
    res1, res2 = await asyncio.gather(t1, t2)

    assert res1 == res2 == 9
    assert call_count == 1

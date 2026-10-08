from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import pytest
import torch

from helpers.walkerTrainingState import (
    RestartableBatchSampler, restore_training_state, save_training_state,
)


def test_sampler_resume_does_not_repeat_prefetched_batches():
    sampler = RestartableBatchSampler(31, 4, 301)
    epoch = list(sampler)
    assert len(epoch) == 7
    assert len({i for batch in epoch for i in batch}) == 28
    sampler.set_step(3)
    assert list(sampler) == epoch[3:]
    sampler.set_step(7)
    assert list(sampler) != epoch
    assert list(sampler) == list(RestartableBatchSampler(31, 4, 302))


def test_interrupted_optimizer_and_randomness_match_uninterrupted(tmp_path):
    torch.manual_seed(9)
    data = torch.randn(31, 3)
    target = torch.randn(31, 1)
    initial = torch.nn.Linear(3, 1)
    identity = {"total_steps": 14, "data": "fixed-fixture"}

    def setup():
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
        return model, optimizer, scheduler

    def run(model, optimizer, scheduler, start, stop):
        sampler = RestartableBatchSampler(31, 4, 301)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, target),
            batch_size=1, shuffle=False, batch_sampler=sampler, num_workers=1,
            persistent_workers=True, prefetch_factor=3, multiprocessing_context="spawn",
            generator=torch.Generator().manual_seed(91301))
        step = start
        while step < stop:
            sampler.set_step(step)
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x) + torch.rand(()) * 0.01
                (prediction - y).square().mean().backward()
                optimizer.step()
                scheduler.step()
                step += 1
                if step % 4 == 0:
                    torch.randn(3, 1024)  # validation SIGReg advances the same RNG
                if step == stop:
                    return

    full = setup()
    torch.manual_seed(100)
    run(*full, 0, 14)
    full_rng = torch.get_rng_state().clone()
    partial = setup()
    torch.manual_seed(100)
    run(*partial, 0, 3)
    path = tmp_path / "optimizer.pt"
    save_training_state(path, model=partial[0], optimizer=partial[1], scheduler=partial[2],
                        step=3, identity=identity, history=[{"step": 3}])
    resumed = setup()
    torch.manual_seed(888)
    step, history = restore_training_state(path, model=resumed[0], optimizer=resumed[1],
                                          scheduler=resumed[2], identity=identity)
    assert step == 3 and history == [{"step": 3}]
    run(*resumed, step, 14)
    for actual, expected in zip(resumed[0].parameters(), full[0].parameters()):
        assert torch.equal(actual, expected)
    assert resumed[2].state_dict() == full[2].state_dict()
    assert torch.equal(torch.get_rng_state(), full_rng)
    for key, state in full[1].state_dict()["state"].items():
        for name, expected in state.items():
            actual = resumed[1].state_dict()["state"][key][name]
            assert torch.equal(actual, expected) if isinstance(expected, torch.Tensor) else actual == expected
    with pytest.raises(ValueError, match="differs"):
        restore_training_state(path, model=resumed[0], optimizer=resumed[1],
                               scheduler=resumed[2], identity={**identity, "data": "changed"})


def test_sampler_works_with_original_dataloader_interface():
    sampler = RestartableBatchSampler(31, 4, 301)
    loader = torch.utils.data.DataLoader(torch.arange(31), batch_size=1, shuffle=False,
                                        batch_sampler=sampler, generator=torch.Generator().manual_seed(90))
    assert len(loader) == 7
    assert [batch.tolist() for batch in loader] == list(sampler)

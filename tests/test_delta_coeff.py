"""Delta coefficient scales δ in forward and is preserved by reparametrization."""

import torch

from models.backbones.delta_convnext import DeltaConvNext
from models.blocks.deltaBlock import DeltaConvnextBlock
from optim.deltaCoeffSchedule import delta_coeff_at_epoch


def test_schedule_hold_then_ramp():
    # Exp 3: 0 through epoch 150, 1 at epoch 250.
    assert delta_coeff_at_epoch(0, 150, 250) == 0.0
    assert delta_coeff_at_epoch(149, 150, 250) == 0.0
    assert delta_coeff_at_epoch(150, 150, 250) == 0.0
    assert abs(delta_coeff_at_epoch(200, 150, 250) - 0.5) < 1e-12
    assert delta_coeff_at_epoch(250, 150, 250) == 1.0
    assert delta_coeff_at_epoch(299, 150, 250) == 1.0


def test_schedule_ramp_from_zero():
    # Exp 2: 0 at epoch 0, 1 at epoch 60.
    assert delta_coeff_at_epoch(0, 0, 60) == 0.0
    assert abs(delta_coeff_at_epoch(30, 0, 60) - 0.5) < 1e-12
    assert delta_coeff_at_epoch(60, 0, 60) == 1.0
    assert delta_coeff_at_epoch(79, 0, 60) == 1.0


def test_schedule_always_one():
    assert delta_coeff_at_epoch(0, 0, 0) == 1.0
    assert delta_coeff_at_epoch(17, 0, 0) == 1.0


def test_coeff_zero_matches_shared_only():
    torch.manual_seed(0)
    with_deltas = DeltaConvNext(useDeltas=True)
    with_deltas.rewire()
    shared_only = DeltaConvNext(useDeltas=False)
    shared_only.rewire()
    shared_only.load_state_dict(
        {k: v for k, v in with_deltas.state_dict().items() if "Delta" not in k},
        strict=False,
    )
    for i in range(with_deltas.stage3_length):
        block = with_deltas.deltifiedStage3[i]
        assert isinstance(block, DeltaConvnextBlock)
        block.setDeltas({name: torch.ones_like(d) for name, d in block.deltas()})
    with_deltas.set_delta_coeff(0.0)
    with_deltas.eval()
    shared_only.eval()
    x = torch.randn(2, 3, 224, 224)
    torch.manual_seed(0)
    y_delta = with_deltas(x)
    torch.manual_seed(0)
    y_shared = shared_only(x)
    assert torch.allclose(y_delta, y_shared, atol=1e-5, rtol=1e-5)


def test_reparametrize_preserves_effective_weights_with_coeff():
    stage3_length = 9
    coeff = 0.5
    model = DeltaConvNext(stage3_length=stage3_length)
    model.rewire()
    model.set_delta_coeff(coeff)
    model.setSharedWeights({name: torch.ones_like(w) for name, w in model.getSharedWeights()})
    for i in range(stage3_length):
        block = model.deltifiedStage3[i]
        block.setDeltas({name: torch.full_like(d, float(i)) for name, d in block.deltas()})

    before = []
    shared_before = dict(model.getSharedWeights())
    for i in range(stage3_length):
        before.append(
            {
                name: shared_before[name] + coeff * delta
                for name, delta in model.deltifiedStage3[i].deltas()
            }
        )

    model.reparametrize()

    shared_after = dict(model.getSharedWeights())
    for i in range(stage3_length):
        for name, delta in model.deltifiedStage3[i].deltas():
            effective = shared_after[name] + coeff * delta
            assert torch.allclose(effective, before[i][name]), f"block {i} {name}"

import torch

from qbs.renderer import render
from qbs.strokes import AuditConfig, adapt, direction_loss, split_stroke, templates


def evaluate(s, t):
    p0, p1, p2 = s[:6].reshape(3, 2)
    return (
        (1 - t[:, None]) ** 2 * p0 + 2 * (1 - t[:, None]) * t[:, None] * p1 + t[:, None] ** 2 * p2
    )


def test_de_casteljau_geometry():
    torch.manual_seed(0)
    s = torch.rand(11)
    a, b = split_stroke(s)
    t = torch.linspace(0, 1, 21)
    torch.testing.assert_close(evaluate(a, t), evaluate(s, t / 2))
    torch.testing.assert_close(evaluate(b, t), evaluate(s, 0.5 + t / 2))


def test_direction_orientation_invariant():
    s = torch.tensor([[0.1, 0.5, 0.5, 0.5, 0.9, 0.5, 1.0, 1.0, 1.0, 0.1, 0.7]])
    field = torch.zeros(16, 16, 3)
    field[:, :, 0] = 1
    field[:, :, 2] = 1
    assert direction_loss(s, field) < 1e-6
    reverse = s.clone()
    reverse[:, :6] = s[:, :6].reshape(-1, 3, 2).flip(1).flatten(1)
    torch.testing.assert_close(direction_loss(s, field), direction_loss(reverse, field))


def test_schedule_capacity_protection_and_finiteness():
    target = torch.ones(24, 24, 3) * 0.5
    s = templates(target, 20, 0)
    s[:5, 10] = 0
    field = torch.zeros(24, 24, 3)
    field[:, :, 0] = 1
    with torch.no_grad():
        canvas, stats = render(s, 24, 24, target=target)
        out, born, mapping, event = adapt(
            s,
            stats,
            target,
            canvas,
            field,
            torch.full((20,), -1000),
            100,
            1,
            AuditConfig(),
            capacity=20,
        )
        assert len(out) == 20 and len(mapping) == 20 and torch.isfinite(out).all()
        assert event["pruned"] > 0
        assert event["split"] - event["pruned"] + event["spawned"] == len(out) - len(s)
        # Protect all edited primitives regardless of their opacity.
        out2, _, _, event2 = adapt(
            s,
            stats,
            target,
            canvas,
            field,
            torch.full((20,), 100),
            100,
            1,
            AuditConfig(),
            capacity=20,
        )
        assert event2 == dict(pruned=0, split=0, spawned=0)
        torch.testing.assert_close(s, out2)


def test_pruning_precedes_splitting_and_counts_actual_edits():
    target = torch.ones(16, 16, 3) * 0.5
    s = templates(target, 10, 3)
    s[:2, 10] = 0
    stats = dict(
        benefit=torch.ones(10),
        purity=torch.full((10,), 0.9),
        area=torch.ones(10) * 10,
        residual=torch.arange(10, 0, -1).float(),
    )
    field = torch.zeros(16, 16, 3)
    field[:, :, 0] = 1
    out, _, _, event = adapt(
        s,
        stats,
        target,
        target,
        field,
        torch.full((10,), -1000),
        100,
        0,
        AuditConfig(max_edit_fraction=0.2),
        capacity=10,
    )
    # The highest-residual split proposal is also deleted due to zero opacity.
    assert event["pruned"] == 2 and event["split"] == 0 and event["spawned"] == 2
    assert len(out) == 10

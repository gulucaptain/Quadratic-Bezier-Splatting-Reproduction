import numpy as np
import torch

from qbs.renderer import contributions, distance, render


def test_distance_against_polynomial_oracle():
    rng = np.random.default_rng(8)
    controls = rng.random((40, 3, 2))
    pixels = rng.random((30, 2))
    actual = distance(torch.tensor(controls), torch.tensor(pixels)).numpy()
    expected = np.zeros_like(actual)
    for i, (p0, p1, p2) in enumerate(controls):
        u, v = p0 - 2 * p1 + p2, p1 - p0
        for j, pixel in enumerate(pixels):
            w = p0 - pixel
            roots = np.roots([u @ u, 3 * u @ v, 2 * v @ v + u @ w, v @ w])
            ts = [0, 1] + [r.real for r in roots if abs(r.imag) < 1e-8 and 0 <= r.real <= 1]
            expected[i, j] = min(np.linalg.norm(p0 + 2 * v * t + u * t * t - pixel) for t in ts)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_degenerate_and_gradient():
    p = torch.tensor([[0.2, 0.3], [0.8, 0.7]], dtype=torch.double)
    c = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]]],
        dtype=torch.double,
    )
    torch.testing.assert_close(distance(c, p)[0], p[:, 1])
    torch.testing.assert_close(distance(c, p)[1], torch.linalg.vector_norm(p - 0.4, dim=-1))
    torch.manual_seed(7)
    c = torch.rand(3, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: distance(c, p), (c,), atol=2e-5)


def test_near_degenerate_reduces_to_segment():
    c = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.0002], [1.0, 0.0]]], dtype=torch.double, requires_grad=True
    )
    p = torch.tensor([[0.3, 0.1]], dtype=torch.double)
    out = distance(c, p)
    torch.testing.assert_close(out, torch.tensor([[0.1]], dtype=torch.double))
    out.sum().backward()
    torch.testing.assert_close(c.grad[:, 1], torch.zeros(1, 2, dtype=torch.double))


def test_float32_nearly_straight_stability():
    rng = np.random.default_rng(81)
    centers = rng.uniform(0.2, 0.8, (40, 2))
    angle = rng.uniform(0, 6.28, 40)
    v = np.stack([np.cos(angle), np.sin(angle)], 1) * 0.2
    normal = np.stack([-np.sin(angle), np.cos(angle)], 1)
    c = np.stack(
        [centers - v, centers + normal * rng.uniform(0.00051, 0.02, (40, 1)), centers + v], 1
    )
    pix = rng.random((30, 2))
    single = distance(torch.tensor(c, dtype=torch.float32), torch.tensor(pix, dtype=torch.float32))
    double = distance(torch.tensor(c, dtype=torch.float64), torch.tensor(pix, dtype=torch.float64))
    torch.testing.assert_close(single.double(), double, atol=2e-6, rtol=0)


def test_tiles_audit_and_composite():
    torch.manual_seed(1)
    s = torch.rand(8, 11, dtype=torch.double)
    s[:, 9] = 0.15
    s.requires_grad_()
    a = render(s, 13, 17, tile=32)
    b = render(s, 13, 17, tile=5)
    torch.testing.assert_close(a, b)
    (ga,) = torch.autograd.grad(a.square().sum(), s)
    (gb,) = torch.autograd.grad(b.square().sum(), s)
    torch.testing.assert_close(ga, gb)
    with torch.no_grad():
        gt = torch.rand_like(a)
        out, stats = render(s, 13, 17, tile=5, target=gt)
        # Improvement telescopes from the black canvas to the final image.
        torch.testing.assert_close(
            (stats["good"] - stats["bad"]).sum(), (gt**2).sum() - ((gt - out) ** 2).sum()
        )
        pix = torch.tensor([[0.5, 0.5]], dtype=torch.double)
        phi = contributions(s, pix, 0.3)
        assert phi.sum() <= 1 + 1e-10


def test_empty_tile_and_checkpoint_gradients():
    s = torch.tensor(
        [[0.1, 0.1, 0.15, 0.13, 0.2, 0.1, 0.9, 0.5, 0.2, 0.02, 0.7]], requires_grad=True
    )
    plain = render(s, 16, 16, tile=8)
    check = render(s, 16, 16, tile=8, recompute=True)
    torch.testing.assert_close(plain, check)
    (g1,) = torch.autograd.grad(plain.sum(), s)
    (g2,) = torch.autograd.grad(check.sum(), s)
    torch.testing.assert_close(g1, g2)

import json

import torch

from .renderer import render

if __name__ == "__main__":
    torch.manual_seed(19)
    s = torch.rand(16, 11)
    s[:, 9] = 0.1
    a = s.clone().requires_grad_()
    b = s.to("mps").requires_grad_()
    x = render(a, 24, 24, tile=16)
    y = render(b, 24, 24, tile=16)
    x.square().mean().backward()
    y.square().mean().backward()
    image_error = float((x - y.cpu()).abs().max().detach())
    gradient_error = float((a.grad - b.grad.cpu()).abs().max())
    assert image_error < 1e-4, (image_error, gradient_error)
    assert gradient_error < 1e-4, (image_error, gradient_error)
    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                mps=True,
                image_max_abs_error=image_error,
                gradient_max_abs_error=gradient_error,
            )
        )
    )

import torch
import torch.nn as nn
import warnings

import torch.optim as optim

from ..attack import Attack


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (DEFALUT : 1e-4)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (DEFALUT : 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (DEFALUT : 1000)
        lr (float): learning rate of the Adam optimizer. (DEFALUT : 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, targeted=False, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: NOT IMPLEMENTED methods in the paper due to time consuming.

        (1) Binary search for c.

        (2) Choosing best L2 adversaries.
    """

    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super(CW, self).__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)

        # f-function in the paper
        def f(x):
            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            return torch.clamp(self._targeted * (j - i), min=-self.kappa)

        w = torch.zeros_like(images).to(self.device)
        w.detach_()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)
        prev = 1e10

        for step in range(self.steps):

            a = 1 / 2 * (nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction="sum")(a, images)
            loss2 = torch.sum(self.c * f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.steps // 10) == 0:
                if cost > prev:
                    warnings.warn("Early stopped because the loss is not converged.")
                    return (1 / 2 * (nn.Tanh()(w) + 1)).detach()
                prev = cost

            # print('- CW Attack Progress : %2.2f %%        ' %((step+1)/self.steps*100), end='\r')

        adv_images = (1 / 2 * (nn.Tanh()(w) + 1)).detach()

        return adv_images

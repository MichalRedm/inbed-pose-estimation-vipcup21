import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Defines the GAN loss which uses either LSGAN or the regular GAN.
    When LSGAN is used, it is basically same as MSELoss,
    but it abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    real_label: torch.Tensor
    fake_label: torch.Tensor

    def __init__(
        self, target_real_label: float = 1.0, target_fake_label: float = 0.0
    ) -> None:
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss: torch.Tensor = self.loss(prediction, target_tensor)
        return loss

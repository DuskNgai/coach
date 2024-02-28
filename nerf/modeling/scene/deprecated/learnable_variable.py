import torch
import torch.nn as nn

def build_learnable_variable(init: float) -> nn.Parameter:
    """Build the learnable variable.
    Args:
        init (float): The initial value of the variable.
    Returns:
        (nn.Parameter): The learnable variable.
    """

    return nn.Parameter(data=torch.tensor([init], dtype=torch.float32), requires_grad=True)

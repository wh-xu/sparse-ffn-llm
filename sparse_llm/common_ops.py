import torch
import torch.nn as nn


def get_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively search for the layers with a certain type in a PyTorch module.

    Base code is from: https://github.com/locuslab/wanda/blob/main/lib/prune.py
    
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
        
    res = {}
    for name1, child in module.named_children():
        res.update(
            get_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def calc_cos_sim(x, y):
    # input shape is [bsz, seq, dim]
    print(x.shape, y.shape)
    norm_x = torch.norm(x, dim=-1).squeeze()
    norm_y = torch.norm(y, dim=-1).squeeze()

    cos_sim = (x @ y) / norm_x / norm_y

    return cos_sim.squeeze()


def svd_reconstruct_with_rank(inp, rank):
    # inp = inp.cpu()
    u, s, v = torch.svd(inp.float())

    recon = torch.mm(
        torch.mm(u[:, :rank], torch.diag(s[:rank])), v[:, :rank].t()
    ).bfloat16()
    return recon

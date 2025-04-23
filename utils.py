import numpy as np
import torch

def CKA(R_1, R_2):
    return (np.linalg.norm(R_1.T*R_2, 'fro') ** 2) / (np.linalg.norm(R_1.T*R_1, 'fro') * (np.linalg.norm(R_2.T*R_2, 'fro')))
    
def l2_dist(model, ref_state):
    device = next(model.parameters()).device
    dist2 = torch.tensor(0.0, device = device)

    sd = ref_state
    for name, p in model.named_parameters():
        if name in sd:
            dist2 += torch.sum((p - sd[name].to(device)).pow(2))
    return dist2

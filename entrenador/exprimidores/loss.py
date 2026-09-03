"""Loss: GJS + focal sin duplicar."""
import torch
import torch.nn.functional as F

def gjs_loss(logits_list):
    """Jensen-Shannon entre vistas."""
    # Promedio de KL
    probs = [F.softmax(l, dim=1) for l in logits_list]
    mean = sum(probs) / len(probs)
    gjs = sum(F.kl_div(p.log(), mean, reduction="batchmean") for p in probs) / len(probs)
    return gjs

def focal_loss(logits, targets, gamma=2.0):
    """Focal para clases desbalanceadas."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1-pt)**gamma * ce).mean()

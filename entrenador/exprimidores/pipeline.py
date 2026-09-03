"""Pipeline: une augment + represent + loss sin duplicar."""
from .augment import augment
from .represent import represent
from .loss import gjs_loss, focal_loss

def exprimir(sequence, model, targets):
    """Exprime un clip: augment -> represent -> loss."""
    seq_aug = augment(sequence)
    views = represent(seq_aug)
    logits = [model(v) for v in views.values()]
    loss = gjs_loss(logits) + focal_loss(logits[0], targets)
    return loss

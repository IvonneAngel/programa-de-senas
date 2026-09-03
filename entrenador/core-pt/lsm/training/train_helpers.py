"""Helpers para train_classifier - split de main y run_epoch."""

def setup_training(args):
    """Configura entrenamiento: parse args, device, data."""
    pass

def run_epoch_split(model, loader, criterion, device, optimizer=None):
    """Ejecuta época: forward, loss, backward."""
    pass

def finalize_training(model, args, best_score):
    """Guarda checkpoint y evalúa."""
    pass

def run_epoch_forward(model, batch, device):
    """Forward pass."""
    pass

def run_epoch_loss(outputs, targets, criterion):
    """Calcula loss."""
    pass

def run_epoch_backward(loss, optimizer):
    """Backward pass."""
    pass

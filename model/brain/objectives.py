"""Shared per-target prediction error for STH-Brain training and semantic evaluation."""
import math
import torch.nn.functional as F


def prediction_error_definition(cfg):
    parameters={'huber_beta':float(cfg.huber_beta),'huber_weight':float(cfg.huber_weight),
                'cosine_weight':float(cfg.cosine_weight)}
    if not all(math.isfinite(value) and value>=0 for value in parameters.values()):
        raise ValueError('STH-Brain prediction-error parameters must be finite and nonnegative')
    if parameters['huber_weight']+parameters['cosine_weight']<=0:
        raise ValueError('STH-Brain prediction error needs at least one positive component weight')
    return {'name':'configured_huber_plus_cosine',**parameters}


def prediction_error(prediction,target,cfg):
    """Return one error per target, using the exact model configuration.

    beta=0 is the explicit SmoothL1/L1 limit supported by PyTorch. Teacher
    stop-gradient and subject aggregation remain responsibilities of callers.
    """
    if prediction.ndim<2 or prediction.shape!=target.shape:
        raise ValueError('STH-Brain prediction and teacher targets must have identical [...,D] shapes')
    definition=prediction_error_definition(cfg)
    prediction=prediction.float();target=target.float()
    huber=F.smooth_l1_loss(prediction,target,reduction='none',beta=definition['huber_beta']).mean(-1)
    if definition['cosine_weight'] == 0:
        return definition['huber_weight']*huber
    cosine=1-F.cosine_similarity(prediction,target,dim=-1)
    return definition['huber_weight']*huber+definition['cosine_weight']*cosine

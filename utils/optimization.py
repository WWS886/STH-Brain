"""Gradient measurements and clipping with explicit optimization-group semantics."""
import torch
import torch.distributed as dist


class ExactCollapseMonitor:
    """Stop sustained exact constants; do not claim a recovery mechanism."""
    def __init__(self,patience):
        if type(patience) is not int or patience<1:
            raise ValueError('Exact-collapse patience must be a positive integer')
        self.patience=patience
        self.consecutive=0

    def observe(self,health):
        if type(health.get('exact_constant')) is not bool:
            raise ValueError('Exact-collapse monitoring requires the measured boolean state')
        self.consecutive=self.consecutive+1 if health['exact_constant'] and health['subject_count']>1 else 0
        return self.consecutive>=self.patience

    def state_dict(self):
        return {'patience':self.patience,'consecutive':self.consecutive}

    def load_state_dict(self,state):
        if set(state)!={'patience','consecutive'} or type(state['patience']) is not int or state['patience']!=self.patience or type(state['consecutive']) is not int or state['consecutive']<0:
            raise ValueError('Exact-collapse monitor resume state mismatch')
        self.consecutive=state['consecutive']


def parameter_groups(model):
    groups = {name: list(module.parameters()) for name, module in
            (('tokenizer', model.student.tokenizer), ('encoder', model.student.encoder))}
    for name in ('roi_temporal','cross_level'):
        if hasattr(model.student,name):groups[name]=list(getattr(model.student,name).parameters())
    groups['predictor']=list(model.predictor.parameters())
    return groups


def vector_norm(tensors):
    parts = [chunk.double().square().sum() for x in tensors
             for chunk in x.detach().reshape(-1).split(1048576)]
    return torch.stack(parts).sum().sqrt() if parts else torch.tensor(0., dtype=torch.float64)


def stable_norm(value):
    return vector_norm([value])


def stable_cosine(first, second):
    products = [(a.double() * b.double()).sum() for a, b in
                zip(first.detach().reshape(-1).split(1048576), second.detach().reshape(-1).split(1048576))]
    return torch.stack(products).sum() / (stable_norm(first).clamp_min(1e-8) * stable_norm(second).clamp_min(1e-8))


def parameter_layer_names(model, params):
    """Match the actual parameter order, including shared module identities."""
    import re
    names={id(p):name for name,p in model.named_parameters()}
    result=[]
    for parameter in params:
        name=names[id(parameter)]
        match=re.match(r'(.+\.(?:layers|local_encoder)\.\d+)\.',name)
        result.append(match.group(1) if match else name.rsplit('.',1)[0])
    return result


def per_layer_norms(model, params, values):
    groups={}
    for name,value in zip(parameter_layer_names(model,params),values):
        groups.setdefault(name,[]).append(value)
    return {name:float(vector_norm(items)) for name,items in groups.items()}


def clip_gradients(model, optim):
    groups = {'student': list(model.student.parameters()),
              'predictor': list(model.predictor.parameters())}
    if optim['clip_mode'] == 'global':
        norms = {k: vector_norm([p.grad for p in v if p.grad is not None]) for k, v in groups.items()}
        total = torch.stack(list(norms.values())).norm()
        torch.nn.utils.clip_grad_norm_(sum(groups.values(), []), optim['grad_clip'], error_if_nonfinite=True)
        factors = {k: min(1., optim['grad_clip'] / (float(total) + 1e-6)) for k in groups}
    elif optim['clip_mode'] == 'student_predictor':
        factors = {}; norms = {}
        for k, params in groups.items():
            limit = optim[k + '_grad_clip']
            norms[k] = torch.nn.utils.clip_grad_norm_(params, limit, error_if_nonfinite=True, foreach=True)
            factors[k] = min(1., limit / (float(norms[k]) + 1e-6))
        total = torch.stack(list(norms.values())).norm()
    else:
        raise ValueError('Unknown clipping mode')
    return {'optimizer/weighted_gradient_norm': float(total),
            **{f'optimizer/{k}_gradient_norm': float(v) for k, v in norms.items()},
            **{f'optimizer/{k}_clip_multiplier': v for k, v in factors.items()}}


def snapshot_parameters(model):
    groups=parameter_groups(model)
    snapshot={name:[p.detach().clone() for p in params] for name,params in groups.items()}
    snapshot['_clipped_gradients']={name:[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params] for name,params in groups.items()}
    return snapshot


def parameter_updates(model, before):
    result = {}
    groups=parameter_groups(model)
    student_groups=tuple(k for k in groups if k!='predictor')
    groups['student']=sum((groups[k] for k in student_groups),[])
    for name, params in groups.items():
        components=student_groups if name=='student' else (name,)
        old_values=sum((before[k] for k in components),[])
        delta=[p.detach()-old for p,old in zip(params,old_values)]
        norm = float(vector_norm(old_values))
        update = float(vector_norm(delta))
        result[f'optimizer/{name}_parameter_norm'] = norm
        result[f'optimizer/{name}_update_norm'] = update
        result[f'optimizer/{name}_update_ratio'] = update / max(norm, 1e-12)
        if '_clipped_gradients' in before:
            gradients=sum((before['_clipped_gradients'][k] for k in components),[])
            gradient=torch.cat([g.float().flatten() for g in gradients])
            actual_delta=torch.cat([d.float().flatten() for d in delta])
            result[f'optimizer/{name}_clipped_gradient_norm']=float(stable_norm(gradient))
            result[f'optimizer/{name}_update_descent_cosine']=float(stable_cosine(actual_delta,-gradient))
            result[f'optimizer/{name}_update_direction_defined']=bool(stable_norm(actual_delta)>0 and stable_norm(gradient)>0)
    layer_rows={}
    for name,params in parameter_groups(model).items():
        for layer,p,old in zip(parameter_layer_names(model,params),params,before[name]):
            values=layer_rows.setdefault(layer,{'old':[],'delta':[]})
            values['old'].append(old);values['delta'].append(p.detach()-old)
    result['optimizer/layers']={name:{'parameter_norm':float(vector_norm(v['old'])),
        'update_norm':float(vector_norm(v['delta'])),
        'relative_update':float(vector_norm(v['delta']))/max(float(vector_norm(v['old'])),1e-12)}
        for name,v in layer_rows.items()}
    return result


def loss_gradient_diagnostics(model, data, amp=False, distributed=True):
    """Separate forward; no .grad writes, steps or RNG changes.

    Use the unwrapped module before a DDP forward. Each rank differentiates
    the gathered subject loss, then averages parameter gradients exactly once,
    matching DDP (including unequal local row counts).
    """
    from model.brain.pretrain import unique_subject_rows, representation_health
    params_by_group = parameter_groups(model)
    params = sum(params_by_group.values(), [])
    device = params[0].device
    devices = [device.index] if device.type == 'cuda' else []
    result = {}
    gradients = {}
    with torch.random.fork_rng(devices=devices):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(**data, objective_distributed=distributed, return_debug=True)
        terms = [('network_prediction', model.objective.prediction_weight),
                 ('roi_prediction', model.objective.prediction_weight*model.objective.roi_prediction_weight),
                 ('variance', model.objective.variance_weight),
                 ('subject_logdet', out['covariance_actual_weight'])]
        for i, (term, weight) in enumerate(terms):
            grads = torch.autograd.grad(out[term + '_loss'], params,
                                        allow_unused=True, retain_graph=i < len(terms) - 1)
            flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                              for p, g in zip(params, grads)])
            if distributed and dist.is_initialized():
                dist.all_reduce(flat)
                flat /= dist.get_world_size()
            if not torch.isfinite(flat).all():
                raise FloatingPointError('Non-finite diagnostic gradient')
            unweighted=flat
            flat=unweighted*weight
            gradients[term] = flat
            result[f'gradient/{term}/weight']=float(weight)
            result[f'gradient/{term}/unweighted_norm']=float(stable_norm(unweighted))
            result[f'gradient/{term}/weighted_norm'] = float(stable_norm(flat))
            result[f'gradient/{term}/loss'] = float(out[term + '_loss'].detach())
            offset = 0
            for name, group in params_by_group.items():
                size = sum(p.numel() for p in group)
                result[f'gradient/{term}/{name}_norm'] = float(stable_norm(flat[offset:offset + size]))
                result[f'gradient/{term}/{name}_unweighted_norm']=float(stable_norm(unweighted[offset:offset+size]))
                offset += size
            student_size=sum(p.numel() for name,group in params_by_group.items() if name!='predictor' for p in group)
            result[f'gradient/{term}/student_weighted_norm']=float(stable_norm(flat[:student_size]))
            result[f'gradient/{term}/student_unweighted_norm']=float(stable_norm(unweighted[:student_size]))
            chunks=list(unweighted.split([p.numel() for p in params]))
            layer_norms=per_layer_norms(model,params,chunks)
            result[f'gradient/{term}/layers']={name:{'unweighted_norm':value,'weighted_norm':value*abs(weight)}
                                               for name,value in layer_norms.items()}
        from itertools import combinations
        for first, second in combinations(gradients,2):
            result[f'gradient/cosine/{first}_{second}'] = float(stable_cosine(gradients[first], gradients[second]))
            result[f'gradient/student_cosine/{first}_{second}']=float(stable_cosine(
                gradients[first][:student_size],gradients[second][:student_size]))
        rows = {'full_student': out['student_full_embeddings'], 'full_student_raw':out['student_full_raw_embeddings'],
                'masked_student': out['student_embeddings']}
        indices = out['target_indices'][:, 0]
        for name, key in (('prediction', 'prediction'), ('teacher_target', 'teacher_target')):
            z = out[key].detach()
            rows[name] = z.new_zeros((len(data['subject_uids']), z.shape[-1])).index_add(0, indices, z)
            rows[name] /= torch.bincount(indices, minlength=len(data['subject_uids']))[:, None]
        for name, z in rows.items():
            health = representation_health(unique_subject_rows(z, data['subject_uids'], distributed))
            result.update({f'diagnostic/{name}/{k}': v for k, v in health.items()})
        result['diagnostic/subjects'] = out['subject_count']
        from utils.stage_health import detailed_health
        result['stage_health']={name:detailed_health(unique_subject_rows(z,data['subject_uids'],distributed))
                               for name,z in out.get('stage_vectors',{}).items()}
        result.update({f'diagnostic/moments/{key}':value for key,value in out['moment_health'].items()})
        result['teacher_target_diagnostics']=out['teacher_target_diagnostics']
        result['roi_routing']=out['roi_routing_diagnostics']
        result['identity_moment_evidence']=dict(zip(('roi','network'),out['identity_moment_evidence']))
        result['mask_diagnostics']=out['mask_diagnostics']
        result['diagnostic/target_and_mask_scope']='rank_local; identity_moments_use_global_unique_subjects_when_distributed'
    return result

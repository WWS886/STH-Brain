"""Layer health measurements during pretraining."""
import torch
import torch.nn.functional as F

LAYER_DIAGNOSTIC_PROTOCOL = 'scan_ln_subject_mean_diagnostics'


@torch.no_grad()
def subject_representations(rows):
    """Compare normalization before and after the observed-visit mean.

    Missing visits never enter an observed-slot mean. Missing values may contain
    arbitrary padding, and are masked before entering the subject reduction.
    """
    from model.brain.content_statistics import subject_slots
    raw, present = subject_slots(rows)
    normalized = {s: [(F.layer_norm(x.float(), (x.shape[-1],)).masked_fill(~m[..., None], 0.), m)
                       for x, m in visits] for s, visits in rows.items()}
    aligned, aligned_present = subject_slots(normalized)
    if not torch.equal(present, aligned_present):
        raise ValueError('Normalization changed observed support')
    legacy = F.layer_norm(raw.float(), (raw.shape[-1],)).masked_fill(~present[..., None], 0.)
    return {'raw_subject_mean': raw, 'scan_ln_subject_mean': aligned,
            'subject_mean_ln_legacy': legacy}, present


def select_stage_subjects(frame, maximum=128, seed=61050):
    """Fixed source-balanced, label-free panel independent of loader/rank order."""
    import hashlib
    records=frame[['subject_uid','dataset']].drop_duplicates()
    groups={}
    for source,rows in records.groupby('dataset'):
        groups[str(source)]=sorted(set(rows.subject_uid.astype(str)),
            key=lambda s:hashlib.sha256(f'{seed}/{s}'.encode()).hexdigest())
    result=[]
    while len(result)<maximum:
        added=False
        for source in sorted(groups):
            if groups[source] and len(result)<maximum:
                result.append(groups[source].pop(0));added=True
        if not added:break
    return set(result)


def layer_location(name):
    """Map diagnostic keys back to the exact executing Python module/layer."""
    import re
    match=re.search(r'layer(\d+)',name)
    if '/tokenizer/' in name:path='model/brain/core.py';module='PhysicalTokenizer'
    elif '/roi_temporal/' in name:path='model/brain/cross_level.py';module='ROITemporalEncoder'
    elif '/network_aggregation/' in name:path='model/brain/pretrain.py';module='BrainTokenizer.summarize_networks'
    elif '/network_encoder/' in name:path='model/sth_brain_encoder.py';module='STHBrainEncoder'
    elif name.startswith('roi_predictor/'):path='model/brain/roi_identity.py';module='ROIPredictor'
    elif name.startswith('network_predictor/'):path='model/sth_brain_predictor.py';module='STHBrainPredictor'
    else:path='model/brain/pretrain.py';module='BrainPretrainer.teacher_target'
    return dict(file=path,module=module,layer=int(match[1]) if match else None,stage=name)


@torch.no_grad()
def build_layer_report(rows, source_by_subject, expected_subjects, *, provenance=None,
                       compression_ratio=.5, compression_minimum_rank=.05):
    import hashlib,json
    from model.brain.content_statistics import subject_slots,observed_center
    from utils.content_health import content_health
    report={};suspects=[];previous={};supports={}
    for name,items in rows.items():
        entry=layer_location(name)
        subjects=sorted(items)
        if set(subjects)!=set(expected_subjects):raise ValueError('Layer panel subject coverage mismatch: '+name)
        values,present=subject_slots(items)
        supports[name]=present
        entry.update(subjects=len(subjects),slots=values.shape[1],width=values.shape[-1],
                     source_counts={s:sum(source_by_subject[u]==s for u in subjects) for s in sorted({source_by_subject[u] for u in subjects})})
        finite=bool(torch.isfinite(values[present]).all());entry['finite']=finite
        flags=[]
        if finite:
            representations,present=subject_representations(items)
            metrics={key:content_health(value,present) for key,value in representations.items()}
            health=metrics['subject_mean_ln_legacy']
            aligned=metrics['scan_ln_subject_mean']
            entry['representations']=metrics
            entry['primary_representation']='scan_ln_subject_mean'
            entry['legacy_top_level_health']=True
            entry['normalization_order']={
                'raw_subject_mean':'observed_visit_mean',
                'scan_ln_subject_mean':'scan_slot_layernorm_then_observed_visit_mean',
                'subject_mean_ln_legacy':'observed_visit_mean_then_slot_layernorm'}
            entry['dtype']='float32_features_float64_centering_and_spectrum'
            entry['slot_subject_counts']=present.sum(0).tolist()
            entry['visit_counts_by_subject']={s:len(items[s]) for s in subjects}
            entry['provenance']=dict(provenance or {})
            centered,counts,_=observed_center(values,present)
            std=(centered.double().square().sum(0)/counts.clamp_min(1)[:,None]).sqrt()
            entry.update(health,raw_feature_std=float(std[counts>=2].mean()) if bool((counts>=2).any()) else 0.)
            if aligned['exact_constant']:flags.append('constant_observed_content')
            prefix=name.rsplit('/',1)[0]
            if entry['layer'] is not None:
                prior=previous.get(prefix)
                if prior and prior[1]>.05 and aligned['capacity_normalized_rank']<prior[1]*.5:
                    flags.append('rank_less_than_half_previous_layer');entry['compared_with']=prior[0]
                previous[prefix]=(name,aligned['capacity_normalized_rank'])
            if aligned['eligible_slots']==0:flags.append('insufficient_observed_support')
        else:flags.append('nonfinite_observed_content')
        entry['diagnostic_flags']=flags
        if flags:suspects.append({**layer_location(name),'flags':flags})
        report[name]=entry
    subjects=sorted(expected_subjects)
    compression={}
    for prefix,end in [('roi_predictor','output'),('network_predictor','output')]+[
            (view+'/roi_temporal','output_norm') for view in ('student_context','student_full','teacher_full')]+[
            (view+'/network_encoder','network_final_norm') for view in ('student_context','student_full','teacher_full')]:
        stack=[(name,entry) for name,entry in report.items()
               if name.rsplit('/',1)[0]==prefix and entry['layer'] is not None and entry['finite']]
        stack.sort(key=lambda item:item[1]['layer'])
        last=prefix+'/'+end
        if not stack or last not in report or not report[last]['finite']:continue
        first=stack[0][0]
        start_health=report[first]['representations']['scan_ln_subject_mean']
        end_health=report[last]['representations']['scan_ln_subject_mean']
        initial=start_health['capacity_normalized_rank'];final=end_health['capacity_normalized_rank']
        same_support=torch.equal(supports[first],supports[last])
        ratio=final/initial if initial>0 else None
        candidate=bool(same_support and initial>compression_minimum_rank and ratio<compression_ratio)
        compression[prefix]={'first_stage':first,'output_stage':last,'rank_ratio':ratio,
                            'same_observed_support':same_support,'candidate':candidate,
                            'first_health':start_health,'output_health':end_health,
                            'raw_std_ratio':report[last]['raw_feature_std']/max(report[first]['raw_feature_std'],1e-12),
                            'consecutive_checks':int(candidate),'needs_review':False}
    comparisons={}
    for predictor,target in (('network_predictor/output','teacher_target/network_ln'),
                              ('roi_predictor/output','teacher_target/roi_ln')):
        if predictor in rows and target in rows:
            pred,pm=subject_representations(rows[predictor]);truth,tm=subject_representations(rows[target])
            common=pm&tm;a=pred['raw_subject_mean'][common];b=truth['raw_subject_mean'][common]
            if a.shape==b.shape and a.numel() and bool(torch.isfinite(a).all()&torch.isfinite(b).all()):
                comparisons[predictor]={'target_stage':target,'observed_subject_slots':int(common.sum()),
                    'subject_slot_mean_squared_error':float((a-b).square().mean()),
                    'subject_slot_mean_cosine':float(F.cosine_similarity(a,b,dim=-1).mean()),
                    'scope':'pooled_subject_slot_comparison_not_token_training_loss',
                    'prediction_spectrum':report[predictor]['representations']['raw_subject_mean'],
                    'target_spectrum':report[target]['representations']['raw_subject_mean']}
    return dict(scope='fixed_validation_panel; diagnostic flags are not causal proof',
                diagnostic_protocol=LAYER_DIAGNOSTIC_PROTOCOL,primary_representation='scan_ln_subject_mean',
                provenance=dict(provenance or {}),compression=compression,prediction_target_comparison=comparisons,
                compression_policy={'ratio_below':compression_ratio,'minimum_reference_rank':compression_minimum_rank},
                content_protocol='observed_subject_content',subject_count=len(subjects),subjects=subjects,
                subjects_sha256=hashlib.sha256(json.dumps(subjects).encode()).hexdigest(),
                layer_count=len(report),layers=report,suspects=suspects)


def annotate_compression_persistence(report, previous, epoch, *, interval=1, patience=3):
    """Only adjacent checks of the same panel/protocol can accumulate warnings."""
    keys=('diagnostic_protocol','subjects_sha256','compression_policy')
    provenance_keys=('source_sha256','panel_sha256','mask_seed','precision')
    compatible=bool(previous and previous.get('epoch')==epoch-interval
                    and all(previous.get(k)==report.get(k) for k in keys)
                    and all(previous.get('provenance',{}).get(k)==report.get('provenance',{}).get(k) for k in provenance_keys))
    for name,entry in report['compression'].items():
        prior=previous.get('compression',{}).get(name,{}) if compatible else {}
        count=1+prior.get('consecutive_checks',0) if entry['candidate'] else 0
        entry.update(consecutive_checks=count,needs_review=count>=patience,
                     patience=patience,decision_scope='diagnostic_only')
    report['epoch']=epoch
    return report


@torch.no_grad()
def stage_vectors(encoded):
    from model.brain.readout import roi_mean_readouts
    from model.brain.roi_identity import diagnostic_vectors
    rows={}
    for name,key in (('tokenizer','roi_local_tokens'),('roi_temporal','roi_temporal_tokens')):
        if key not in encoded:continue
        pooled=roi_mean_readouts({**encoded,'roi_tokens':encoded[key]})
        raw=pooled['per_roi'].float();valid=pooled['valid']
        rows[name+'_raw']=raw.masked_fill(~valid[...,None],0).flatten(1)
        rows[name+'_ln']=F.layer_norm(raw,(raw.shape[-1],)).masked_fill(~valid[...,None],0).flatten(1)
        # Per-ROI temporal variation tests information lost by time averaging.
        second=roi_mean_readouts({**encoded,'roi_tokens':encoded[key].float().square()})['per_roi']
        rows[name+'_temporal_std']=(second-raw.square()).clamp_min(0).sqrt().flatten(1)
    from model.brain.readout import network_mean_readouts
    for name,tokens in encoded.get('network_stage_tokens',{}).items():
        pooled=network_mean_readouts(tokens,encoded['token_valid'],encoded['token_network_ids'],encoded['num_networks'])
        raw,present=pooled['raw'],pooled['valid']
        rows[name+'_raw']=raw.masked_fill(~present[...,None],0).flatten(1)
        rows[name+'_ln']=F.layer_norm(raw,(raw.shape[-1],)).masked_fill(~present[...,None],0).flatten(1)
    rows.update(diagnostic_vectors(encoded))
    return {key:value.detach() for key,value in rows.items()}


@torch.no_grad()
def detailed_health(values):
    from model.brain.pretrain import representation_health
    values=values.float();health=representation_health(values)
    centered=values-values.mean(0);gram=centered@centered.T
    energy=gram.diagonal().sum();sq=gram.square().sum()
    std=centered.square().mean(0).sqrt()
    eig=torch.linalg.eigvalsh(gram).clamp_min(0)
    return {**health,'subjects':len(values),'dimension':values.shape[-1],
            'capacity':min(max(len(values)-1,0),values.shape[-1]),
            'top1_energy':float(eig[-1]/eig.sum().clamp_min(1e-12)),
            'entropy_rank':health['normalized_effective_rank']*values.shape[-1],
            'participation_rank':float(energy.square()/sq.clamp_min(1e-12)) if energy>1e-12 else 0.,
            'centered_energy':float(energy), 'std_p05':float(torch.quantile(std,.05)),
            'std_median':float(std.median())}

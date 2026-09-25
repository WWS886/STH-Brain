"""Sequence-only inputs and complete, subject-balanced distributed epochs."""
import random
from torch.utils.data import Sampler
from dataset.sth_brain_collate import STHBrainCollate


def make_dataset(cfg,split):
    from dataset.registered_splits import RegisteredDataset,INTERFACE
    from pathlib import Path
    d=cfg['data']
    if d.get('interface') != INTERFACE:
        raise ValueError('Pretraining requires the registered subject interface')
    return RegisteredDataset(d['manifest'],split,cache_dir=Path(d['sequence_cache_receipt_path']).parent)


def make_collate(cfg,epoch=0,seed=None):
    m=cfg['mask'];model=cfg['model']
    c=STHBrainCollate(roi_network_path=model['atlas_path'],block_seconds=model['block_seconds'],min_block_coverage=model['min_block_coverage'],
        physical_grid_step_seconds=model['physical_grid_step_seconds'],max_interpolation_gap_seconds=model['max_interpolation_gap_seconds'],
        target_coverage=m['target_coverage'],target_coverage_start=m['target_coverage'],coverage_curriculum_epochs=0,
        exclusion_seconds=m['exclusion_seconds'],min_exclusion_blocks=m['min_exclusion_blocks'],min_context_fraction=m['min_context_fraction'],
        max_mask_resamples=m['max_resamples'],seed=cfg['runtime']['seed'] if seed is None else seed,view_id=0,support_aware_masks=True,minimum_valid_networks=2,minimum_supported_mask_tokens=16)
    c.mask_strategy=m.get('strategy','union')
    c.set_epoch(epoch)
    return c


class CompleteSubjectBatchSampler(Sampler):
    """Every subject exactly once per epoch; choose one scan, never pad with duplicates.

    Merge a very small remainder into the preceding batch so every rank has >=1
    row. Most batches retain global batch_size, and the last may be unequal.
    """
    def __init__(self,subjects,batch_size,world_size=1,rank=0,seed=42):
        self.groups={}
        for i,s in enumerate(subjects):self.groups.setdefault(str(s),[]).append(i)
        self.subjects=sorted(self.groups);self.batch_size=batch_size;self.world=world_size;self.rank=rank;self.seed=seed;self.epoch=0
        if len(self.subjects)<world_size or batch_size<world_size:raise ValueError('Insufficient subjects for distributed moments')
    def set_epoch(self,e):self.epoch=e
    def global_batches(self):
        rng=random.Random(self.seed+self.epoch*1000003);subjects=self.subjects.copy();rng.shuffle(subjects)
        indices=[rng.choice(self.groups[s]) for s in subjects]
        chunks=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        if len(chunks)>1 and len(chunks[-1])<self.world:chunks[-2].extend(chunks.pop())
        return chunks
    def __iter__(self):
        for chunk in self.global_batches():yield chunk[self.rank::self.world]
    def __len__(self):return len(self.global_batches())


class SubjectEvaluationSampler(Sampler):
    """Disjoint, unpadded evaluation shards; all visits stay with their subject.

    Assign in first-occurrence order, balancing scan counts deterministically.
    The optional limit is a GLOBAL prefix, identical to serial smoke evaluation.
    """
    def __init__(self, subjects, world_size=1, rank=0, max_scans=0):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid evaluation rank/world size')
        groups = {}
        for index, subject in enumerate(subjects):
            if max_scans and index >= max_scans:break
            groups.setdefault(str(subject), []).append(index)
        loads = [0] * world_size; indices = []
        for rows in groups.values():
            owner = min(range(world_size), key=lambda r: (loads[r], r))
            loads[owner] += len(rows)
            if owner == rank:indices.extend(rows)
        self.indices = sorted(indices)

    def __iter__(self):return iter(self.indices)
    def __len__(self):return len(self.indices)


def model_inputs(batch,device):
    keys=('block_values','block_sample_valid','block_sample_offsets','block_valid','block_center_seconds','block_duration_seconds','context_mask','target_mask','exclusion_mask')
    result={k:batch['sth_brain_'+k].to(device,non_blocking=True) for k in keys}
    result['tr_seconds']=batch['tr'].to(device,non_blocking=True)
    result['roi_valid_mask']=batch['roi_valid_mask'].to(device,non_blocking=True)
    result['target_type']=batch['sth_brain_target_type'].to(device,non_blocking=True)
    result['source_names']=list(batch.get('dataset',['unknown']*len(batch['subject_uid'])))
    result['subject_uids']=list(batch['subject_uid'])
    return result

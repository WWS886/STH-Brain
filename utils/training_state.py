"""Record successful pretraining progress and completion."""
from pathlib import Path
from datetime import datetime, timezone
import os
from settings.pretrain import ALGORITHM_VERSION
from utils.pretrain_contract import atomic_json


def finalize_training_state(run, epoch, cfg, source, *, smoke=False):
    run = Path(run)
    target_epochs = cfg['optim']['epochs']
    completed_epochs = epoch >= target_epochs
    status = 'smoke_complete' if smoke else ('complete' if completed_epochs else 'paused')
    state = {'status': status, 'epoch': epoch, 'target_epochs': target_epochs, 'pid': os.getpid(),
             'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
             'algorithm_version': ALGORITHM_VERSION, 'source_sha256': source['sha256'],
             'completed_epochs': completed_epochs}
    atomic_json(run / 'status.json', state)
    if completed_epochs and not smoke:
        atomic_json(run / 'training_complete.json', {
            'algorithm_version': ALGORITHM_VERSION, 'epoch': epoch,
            'source_sha256': source['sha256'],
            'status': status, 'smoke': smoke, 'test_evaluated': False})
    return state

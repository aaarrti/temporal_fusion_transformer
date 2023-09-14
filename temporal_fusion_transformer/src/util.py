from __future__ import annotations

import os
from absl import logging


def supports_mixed_precision() -> bool:
    cc = os.popen("nvidia-smi --query-gpu=compute_cap --format=csv").readlines()
    if len(cc) == 0:
        logging.error("Could not verify CUDA compute capability")
        return False
    
    cc = cc[-1]
    
    if float(cc) >= 7.5:
        logging.info(f"Compute capability {cc} -> mixed precision OK")
        return True
    else:
        logging.info(f"Compute capability {cc} -> mixed precision NOT OK")
        return False

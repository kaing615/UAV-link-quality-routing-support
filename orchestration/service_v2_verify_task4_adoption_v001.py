import json
from pathlib import Path
import sys
root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
sys.path[:0]=[str(root/'orchestration'),str(root/'source-runtime-v003')]
import service_v2_parent_phase_v004 as parent
study=root/'reports/uav_service_prevention_v2_2026-11-12'
print(json.dumps(parent.verify_adoption(study/'engineering/task4-parent-adoption-v001.json',study,root/'source-runtime-v003'),indent=2))

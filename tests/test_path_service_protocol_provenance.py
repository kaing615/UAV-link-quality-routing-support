"""Only byte-exact LF/CRLF protocol drift is accepted by offline evaluation."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis.evaluate_path_service import verify_protocol_representation


def sha(data):return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('source,copied',[
    (b'{\r\n  "a": 1\r\n}',b'{\n  "a": 1\n}'),
    (b'{\n  "a": 1\n}',b'{\r\n  "a": 1\r\n}'),
    (b'{\n  "a": 1\n}',b'{\n  "a": 1\n}'),
])
def test_accepts_only_reconstructed_source_hash(tmp_path,source,copied):
    path=tmp_path/'protocol.json';path.write_bytes(copied)
    result=verify_protocol_representation(path,sha(source))
    assert result['original_frozen_protocol_sha256']==sha(source)
    assert result['copied_run_protocol_sha256']==sha(copied)
    assert result['validation']==('exact_bytes' if source==copied else 'verified_LF_CRLF_only')


@pytest.mark.parametrize('copied',[
    b'{\n  "a": 2,\n  "b": 3\n}',  # changed setting
    b'{\n "a": 1,\n "b": 3\n}',  # equivalent JSON, changed indentation
    b'{\n  "b": 3,\n  "a": 1\n}',  # equivalent JSON, changed key order
    b'{\n  "a": 1,\n  "b": 3\n}\n',  # added trailing newline
    b'{\r  "a": 1,\r  "b": 3\r}',  # unsupported lone CR
])
def test_rejects_other_tampering_even_semantically_equivalent(tmp_path,copied):
    source=b'{\r\n  "a": 1,\r\n  "b": 3\r\n}'
    path=tmp_path/'protocol.json';path.write_bytes(copied)
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_protocol_representation(path,sha(source))


def test_actual_pilot_v2_matches_frozen_bytes_by_line_endings():
    root=Path(__file__).resolve().parents[1]
    run=root/'reports/uav_path_service_study_2026-11-11/pilot-v2-raw/pilot-v2'
    if not run.is_dir():pytest.skip('Actual downloaded pilot-v2 is required for this integration fixture')
    source=root/'.superpowers/sdd/2026-11-11-routing-service/task-3-frozen/pilot.json'
    raw=(run/'protocol.json').read_bytes();frozen=source.read_bytes()
    assert frozen.replace(b'\r\n',b'\n')==raw
    result=verify_protocol_representation(run/'protocol.json',sha(frozen))
    assert result['validation']=='verified_LF_CRLF_only'

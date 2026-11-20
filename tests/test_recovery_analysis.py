"""Protect existing recovery evaluation evidence from accidental reruns."""
from types import SimpleNamespace

import pytest
import scripts.analysis.evaluate_recovery as analysis


@pytest.mark.parametrize('existing_file', [False, True])
def test_evaluation_refuses_existing_evidence_before_launch(tmp_path, monkeypatch, existing_file):
    binary = tmp_path / 'simulator'
    binary.write_bytes(b'stub')
    output = tmp_path / 'evaluation'
    if existing_file:
        output.write_text('existing evidence', encoding='utf-8')
        evidence = output
    else:
        output.mkdir()
        evidence = output / 'summary.json'
        evidence.write_text('existing evidence', encoding='utf-8')

    def unexpected_launch(*args):
        raise AssertionError('reused evaluation output reached simulator preparation')

    monkeypatch.setattr(analysis, 'make_launcher', unexpected_launch)
    args = SimpleNamespace(binary=binary, output=output, steps=120, wsl=None, scenarios=1, model=[])
    with pytest.raises(ValueError, match='evaluation output'):
        analysis.run(args)
    assert evidence.read_text(encoding='utf-8') == 'existing evidence'

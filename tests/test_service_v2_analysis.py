import pytest
from copy import deepcopy
import hashlib
import numpy as np
from src.control.service_v2.config import POLICIES
from tests.service_v2_support import service_frames,episode_from_frames
from src.control.service_v2.analysis import paired_effects, episode_endpoints, binary_scores
from tests.test_service_v2_protocol import frozen
from tests.test_service_v2_study import collection


def rows():
    return [dict(regime=regime,context='gradual',layout_seed=i,radio_seed=i+100,
                 policy=p,status='ok',actual_udp_pdr=.8,total_relay_travel_m=10.)
        for regime,n in [('n4',2),('n5',5)] for i in range(n)
        for p in POLICIES]


def test_equal_policies_have_zero_effect():
    result=paired_effects(rows(),'service-v2-no-forecast')
    assert result['pooled']['pdr_difference']==0
    assert result['pooled']['pdr_ci95']==[0,0]


def test_equal_regime_means_and_missing_sensitivity():
    r=rows()
    for row in r:
        if row['regime']=='n4' and row['policy']=='service-v2': row['actual_udp_pdr']=1.
    assert paired_effects(r,'service-v2-no-forecast')['pooled']['pdr_difference']==pytest.approx(.1)
    missing=next(row for row in r if row['policy']=='service-v2')
    missing['status']='technical_error'; missing['actual_udp_pdr']=None
    s=paired_effects(r,'service-v2-no-forecast')['pooled']['missing_pdr_sensitivity']
    assert s[0] < s[1]


def eight_role_blocks():
    endpoint=episode_endpoints(episode_from_frames(service_frames([60]*240)))
    result=[]
    for context in ('static','gradual'):
        for regime in ('n4','n5'):
            for i in range(3):
                seed=(100 if context=='gradual' else 1000)+(0 if regime=='n4' else 10000)+i
                v2=.70+.05*i
                for role in POLICIES:
                    pdr=v2 if role=='service-v2' else .50 if role=='hold' else 1.-v2 if role=='reactive' else .60
                    result.append(dict(context=context,regime=regime,layout_seed=seed,radio_seed=seed+100000,
                        policy=role,status='ok',actual_udp_pdr=pdr,total_relay_travel_m=10.+i,
                        endpoints=deepcopy(endpoint)))
    return result


def test_primary_missing_uncompared_role_excludes_the_entire_block():
    r=eight_role_blocks()
    r=[row for row in r if not (row['context']=='gradual' and row['regime']=='n4' and row['layout_seed']==100 and row['policy']=='geometry')]
    result=paired_effects(r,'hold')['regimes']['n4']
    assert result['planned_pairs']==3 and result['complete_pairs']==2
    assert result['pdr_difference']==pytest.approx(.275)
    # HOLD and v2 are observed in all three planned blocks; their PDR bounds
    # still describe every planned pair, not just the complete-block estimate.
    assert result['missing_pdr_sensitivity']==pytest.approx([.25,.25])
    assert result['block_bootstrap']['incomplete_blocks'][0]['unavailable_roles']==['geometry']


def test_all_role_contrasts_use_one_shared_draw_matrix_per_stratum(monkeypatch):
    from src.control.service_v2.analysis import _effect_tables, BOOTSTRAP_SAMPLES
    draws=[]
    class SpecifiedDraws:
        def integers(self,low,high,size):
            # Independently specified resamples: alternating lowest/highest
            # complete scenario for the four context/regime strata.
            draw=np.full(size,0 if len(draws)%2==0 else high-1,dtype=np.int64)
            draws.append(draw)
            return draw
    monkeypatch.setattr(np.random,'default_rng',lambda seed:SpecifiedDraws())
    primary,secondary=_effect_tables(eight_role_blocks())
    assert len(draws)==4
    for index,(context,regime) in enumerate((('static','n4'),('static','n5'),('gradual','n4'),('gradual','n5'))):
        block=secondary[context][regime]['block_bootstrap']
        assert block['roles_kept_together']==list(POLICIES)
        assert block['draw_shape']==[BOOTSTRAP_SAMPLES,3]
        assert block['draw_indices_sha256']==hashlib.sha256(draws[index].astype('<i8').tobytes()).hexdigest()
        expected=.20 if index%2==0 else .30
        effects=secondary[context][regime]['contrasts']
        assert effects['hold']['actual_udp_pdr']['descriptive_ci95']==pytest.approx([expected,expected])
        assert effects['reactive']['actual_udp_pdr']['descriptive_ci95']==pytest.approx([2*expected,2*expected])
        if context=='gradual':
            assert primary['hold']['regimes'][regime]['block_bootstrap']==block
            assert primary['reactive']['regimes'][regime]['block_bootstrap']==block
            assert primary['hold']['regimes'][regime]['pdr_ci95']==pytest.approx([expected,expected])


def test_entirely_missing_role_cannot_shrink_required_role_set():
    from src.control.service_v2.analysis import secondary_effects
    r=[row for row in eight_role_blocks() if row['policy']!='geometry']
    primary=paired_effects(r,'hold')
    assert primary['pooled']['pdr_difference'] is None
    assert primary['pooled']['missing_pdr_sensitivity']==pytest.approx([.25,.25])
    for group in secondary_effects(r).values():
        for result in group.values():
            assert result['all_role_complete_pairs']==0
            assert result['roles_kept_together']==list(POLICIES)
            assert all(b['unavailable_roles']==['geometry'] for b in result['block_bootstrap']['incomplete_blocks'])


def test_no_event_is_right_censored():
    e=episode_from_frames(service_frames([60]*240))
    row=episode_endpoints(e)
    assert row['events']['degraded']['onset_step'] is None
    assert row['events']['degraded']['right_censored']
    assert row['events']['degraded']['restricted_event_free_seconds']==240


def test_undefined_precision_and_pr_auc_are_null():
    result=binary_scores([0,0],[.1,.2])
    assert result['precision'] is None and result['pr_auc'] is None


def test_complete_analysis_and_exclusive_output(collection,tmp_path):
    from src.control.service_v2.analysis import analyze_collection
    from src.control.service_v2.reporting import _verified_input
    protocol,root=collection
    result=analyze_collection(protocol,root,tmp_path/'analysis')
    assert result['complete'] and result['ok_cells']==32
    assert len(result['rows'])==32 and result['forecasts'] is None
    _verified_input(tmp_path/'analysis','analysis.json','service-v2-analysis-v1')
    before=(tmp_path/'analysis/analysis.json').read_bytes()
    with pytest.raises(FileExistsError): analyze_collection(protocol,root,tmp_path/'analysis')
    assert (tmp_path/'analysis/analysis.json').read_bytes()==before


def test_readonly_revalidation_after_explicit_transfer(collection,tmp_path):
    import shutil
    from src.control.service_v2.study import validate_collection
    protocol,root=collection
    copied=tmp_path/'copied-frozen'; shutil.copytree(protocol.parent,copied)
    raw=tmp_path/'copied-raw'; shutil.copytree(root,raw)
    result=validate_collection(copied/'protocol.json',raw)
    assert result['complete'] and result['ok_cells']==32


def test_warning_matching_uses_shortest_horizon_and_keeps_censoring():
    import numpy as np
    from src.control.service_v2.prediction import ServicePredictor
    from src.control.service_v2.analysis import forecast_scores
    class SyntheticPredictor(ServicePredictor):
        # Hand-specified coefficients solely exercise analysis; no fitted artifact
        # or simulation is produced, and public artifact loading is tested elsewhere.
        def __init__(self):
            models={}
            for h in ('10','20'):
                weights=[0.]*25
                if h=='20': weights[9]=-100.
                models[h]=dict(mean=[0.]*25,scale=[1.]*25,weights=weights,
                    intercept=-10. if h=='10' else 24.3,calibration=[0.,1.])
            self._artifact={'models':models}
        def validate_scenarios(self,scenarios): pass
    frames=service_frames([60]*60+[0]*20+[60]*160)
    for f in frames:
        f['positions'][0][0]=300.+min(f['step'],60)
    e=episode_from_frames(frames,phase='formal')
    result=forecast_scores([e],SyntheticPredictor())
    assert result['warnings'] and all(w['horizon']==20 for w in result['warnings'])
    assert result['horizons']['20']['censored_rows']>0
    assert result['detected_event_numerator']==1
    assert result['event_sensitivity']==1
    assert all(s['kind']=='shadow_forecast_opportunity' for s in result['shadow_opportunities'])

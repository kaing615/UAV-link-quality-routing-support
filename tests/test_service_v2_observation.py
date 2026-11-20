from src.control.service_v2.config import ServiceV2Config
from src.control.service_v2.observation import ServiceHistory
from tests.service_v2_support import fixture_config, service_frames, fail_route, set_window_counts


def states(frames):
    h = ServiceHistory(fixture_config(), ServiceV2Config())
    return [h.update(f) for f in frames]


def test_seven_full_windows_arm():
    h = ServiceHistory(fixture_config(), ServiceV2Config())
    states = [h.update(f) for f in service_frames([60] * 7)]
    assert [s.ever_armed for s in states] == [False] * 7 + [True]
    assert states[-1].first_armed_step == 7
    assert states[-1].forecast_eligible
import pytest
from dataclasses import replace



def test_sum_ratios_delayed_arrivals_and_source_injection():
    fs = service_frames([60, 30, 60, 120, 60, 90, 90], [60, 30, 60, 60, 60, 60, 60])
    ss = states(fs)
    assert ss[5].ratios['A5'] == pytest.approx(330 / 270)
    assert ss[7].ratios['A1'] == 1.5
    assert ss[7].ever_armed
    set_window_counts(fs, source=[0] * 7)
    s = states(fs)[-1]
    assert s.ratios['X5'] == 0
    assert s.ratios['Q5'] is None
    assert not s.healthy_evidence and not s.ever_armed


def test_confirmations_have_onsets_and_latched_restoration_until_recovered():
    ss = states(service_frames([60] * 7 + [0] * 7 + [60] * 7))
    assert ss[8].activation_context == 'incipient_delivery_decline'
    assert not ss[9].degraded_confirmed
    assert ss[10].degraded_confirmed and ss[10].outage_confirmed
    assert ss[10].event_onsets['degraded'] == (8,)
    assert ss[10].event_confirmations['degraded'] == (10,)
    assert not ss[11].severe_confirmed and ss[12].severe_confirmed
    assert ss[20].mode == 'RESTORE' and not ss[20].degraded_confirmed
    assert ss[21].mode == 'READY' and ss[21].recovered
    assert ss[21].first_armed_step == 7


def test_startup_and_route_only_restoration_are_not_prevention():
    startup = states(service_frames([0] * 7 + [60] * 7))
    assert startup[3].mode == 'RESTORE'
    assert startup[3].activation_context == 'startup_restoration'
    assert not startup[3].ever_armed
    assert startup[14].first_armed_step == 14
    fs = service_frames([60] * 17)
    for f in fs[8:11]:
        fail_route(f)
    ss = states(fs)
    assert ss[10].route_failure_confirmed and not ss[10].degraded_confirmed
    assert ss[10].activation_context == 'routing_restoration'
    assert ss[12].mode == 'RESTORE'
    assert ss[13].mode == 'READY'


@pytest.mark.parametrize('damage', ['offers', 'duration', 'missing', 'gap'])
def test_unknown_resets_history_and_keeps_latches(damage):
    fs = service_frames([60] * 7 + [0] * 3 + [60] * 12)
    if damage == 'offers':
        fs[11]['service_observation']['window']['offered_packets'] = 29
        set_window_counts(fs)
    elif damage == 'duration':
        fs[11]['service_observation']['window_start_s'] -= .01
    elif damage == 'missing':
        del fs[11]['service_observation']
    else:
        del fs[11]
    ss = {s.step: s for s in states(fs)}
    invalid_step = 12 if damage == 'gap' else 11
    assert ss[invalid_step].mode == 'UNKNOWN'
    assert ss[invalid_step].ever_armed
    assert not ss[invalid_step].valid_history
    assert ss[invalid_step + 4].mode == 'UNKNOWN'
    assert ss[invalid_step + 5].mode == 'RESTORE'
    assert ss[invalid_step + 7].mode == 'READY'


def test_marginal_service_suppresses_new_forecast_without_restoration():
    ss = states(service_frames([60] * 7 + [40]))
    assert ss[-1].ratios['A5'] == pytest.approx(280 / 300)
    assert ss[-1].mode == 'READY'
    assert ss[-1].activation_context == 'marginal_service'
    assert not ss[-1].forecast_eligible


def test_state_never_uses_future_mission_config_or_frame_metadata():
    fs = service_frames([60] * 7)
    alternate = replace(fixture_config(), mission_start_step=999, mission_speed=7.)
    h = ServiceHistory(alternate, ServiceV2Config())
    fs[-1]['future_mission_start_step'] = -123
    fs[-1]['future_endpoint_positions'] = [[9999, 9999, 9999]]
    ss = [h.update(f) for f in fs]
    assert ss == states(service_frames([60] * 7))


def test_repeated_initial_frame_is_unknown_and_cannot_reset_arming_latch():
    h = ServiceHistory(fixture_config(), ServiceV2Config())
    fs = service_frames([60] * 7)
    for f in fs:
        h.update(f)
    replay = h.update(fs[0])
    assert replay.mode == 'UNKNOWN'
    assert replay.ever_armed and replay.first_armed_step == 7

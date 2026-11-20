"""Hand-specified measurements; fixture identities are not research seeds."""
import hashlib
import json

from src.control.prevention.config import PreventionConfig


def fixture_config():
    return PreventionConfig(num_uavs=4, destination=3, comm_range=240,
        initial_spacing_fraction=.15, x_max=1800, steps=240,
        mission_start_step=30, mission_speed=1)


def service_frames(arrivals, offers=None):
    offers = [60] * len(arrivals) if offers is None else list(offers)
    if len(offers) != len(arrivals):
        raise ValueError('Fixture window lengths differ')
    frames, totals = [], [0, 0, 0]
    for step, (arrival, offer) in enumerate([(0, 0), *zip(arrivals, offers)]):
        window = dict(offered_packets=offer, source_tx_packets=offer, sink_rx_packets=arrival)
        totals = [a + b for a, b in zip(totals, window.values())]
        frames.append(dict(version=1, step=step,
            positions=[[300., 600., 100.], [336., 600., 100.], [372., 600., 100.], [408., 600., 100.]],
            velocities=[[0., 0., 0.] for _ in range(4)], travel=[0.] * 4,
            rx_ratio=[[float(a != b) for b in range(4)] for a in range(4)],
            rssi=[[-70. if a != b else 0. for b in range(4)] for a in range(4)],
            rssi_mask=[[a != b for b in range(4)] for a in range(4)],
            components=1, source_dest_connected=True, olsr_route_usable=True,
            connected_streak=step, safety_rejected=False, terminal=False,
            service_observation=dict(schema='service-observation-v1', sim_time_s=11. + step,
                window_start_s=0. if step == 0 else 10. + step, window_end_s=11. + step,
                flow=dict(source=0, destination=3, destination_port=9000, protocol='udp'),
                window=window, cumulative=dict(zip(window, totals)),
                route=dict(nodes=[0, 3], success=True, reason='success', failing_node=None),
                route_probe_hops=[dict(**{'from': 0, 'to': 3}, rx_ratio_forward=1., rx_ratio_reverse=1.)])))
    return frames


def episode_from_frames(frames, *, identity='fixture', phase='train', policy='hold',
                        layout_seed=11, radio_seed=22, context='gradual'):
    from src.control.service_v2.prediction import ServiceEpisode
    digest = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
    return ServiceEpisode(identity, phase, policy, 'n4-.15', context, layout_seed, radio_seed,
                          fixture_config(), frames, {'fixture-source': 'a' * 64, 'fixture-binary': 'b' * 64}, digest)


def set_window_counts(frames, *, source=None):
    """Reconcile counters after deliberately changing fixture windows."""
    totals = dict(offered_packets=0, source_tx_packets=0, sink_rx_packets=0)
    for f in frames:
        s = f['service_observation']
        if source is not None and f['step']:
            s['window']['source_tx_packets'] = source[f['step'] - 1]
        for k in totals:
            totals[k] += s['window'][k]
        s['cumulative'] = dict(totals)


def fail_route(frame):
    s = frame['service_observation']
    s['route'] = dict(nodes=[0], success=False, reason='missing_entry', failing_node=0)
    s['route_probe_hops'] = []


def synthetic_partition(phase, groups):
    """Deterministic measurement blocks, with both contexts and policies.

    These fixture-only IDs never allocate or collect research scenarios.
    Each decline begins at 61; eligible origins precede it, then recovery
    yields a second independent span of healthy measurements in each trace.
    """
    base = 100 if phase == 'train' else 1000
    episodes = []
    for group in range(groups):
        for context in ('static', 'gradual'):
            for policy in ('hold', 'geometry'):
                frames = service_frames([60] * 60 + [0] * 20 + [60] * 35)
                # Measured geometry and velocities vary causally within blocks.
                for frame in frames:
                    j = frame['step']
                    offset = (j / 10. if context == 'gradual' else 0.) + group
                    frame['positions'][0][1] -= offset
                    frame['positions'][3][1] += offset
                    if context == 'gradual':
                        frame['velocities'][0][1], frame['velocities'][3][1] = -.1, .1
                    if policy == 'geometry':
                        frame['travel'][1] = frame['travel'][2] = j * .1
                episodes.append(episode_from_frames(frames,
                    identity=f'fixture-{phase}-{group}-{context}-{policy}', phase=phase, policy=policy,
                    layout_seed=base + group, radio_seed=base + 100 + group, context=context))
    return episodes

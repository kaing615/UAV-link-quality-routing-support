import math
import config


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_rssi(distance: float) -> float:
    tx_power_dbm = float(getattr(config, "TX_POWER_DBM", 20.0))
    reference_loss_db = float(getattr(config, "REFERENCE_PATH_LOSS_DB", 40.0))
    path_loss_exponent = float(getattr(config, "PATH_LOSS_EXPONENT", 2.2))

    d = max(distance, 1.0)
    path_loss_db = reference_loss_db + 10.0 * path_loss_exponent * math.log10(d)
    return tx_power_dbm - path_loss_db


def estimate_snr(rssi: float) -> float:
    noise_floor_dbm = float(getattr(config, "NOISE_FLOOR_DBM", -90.0))
    return rssi - noise_floor_dbm


def estimate_delay(
    distance: float,
    relative_speed: float,
    connected: int,
    load_factor: float,
) -> float:
    propagation_ms = distance / 300_000_000.0 * 1000.0
    base_processing_ms = float(getattr(config, "BASE_DELAY_MS", 2.0))
    mobility_penalty_ms = 0.08 * relative_speed
    contention_penalty_ms = 4.0 * load_factor
    disconnected_penalty_ms = (
        float(getattr(config, "DISCONNECTED_DELAY_MS", 50.0)) if not connected else 0.0
    )

    delay = (
        base_processing_ms
        + propagation_ms
        + mobility_penalty_ms
        + contention_penalty_ms
        + disconnected_penalty_ms
    )
    return round(delay, 4)


def estimate_packet_loss(snr: float, connected: int, load_factor: float) -> float:
    if not connected:
        return 1.0

    if snr >= 25.0:
        loss = 0.01
    elif snr >= 18.0:
        loss = 0.03
    elif snr >= 12.0:
        loss = 0.08
    elif snr >= 8.0:
        loss = 0.18
    else:
        loss = 0.35

    loss += 0.10 * load_factor
    return round(min(max(loss, 0.0), 1.0), 4)


def estimate_throughput(
    snr: float,
    packet_loss: float,
    connected: int,
    load_factor: float,
) -> float:
    max_rate_mbps = float(getattr(config, "MAX_THROUGHPUT_MBPS", 100.0))

    if not connected:
        return 0.0

    snr_eff = min(max(snr, 0.0), 30.0) / 30.0
    throughput = max_rate_mbps * snr_eff * (1.0 - packet_loss) * (1.0 - 0.35 * load_factor)
    return round(max(throughput, 0.0), 4)


def estimate_p_stable(snr: float, packet_loss: float, delay: float, connected: int) -> float:
    if not connected:
        return 0.0

    snr_score = clamp01((snr - 5.0) / 20.0)
    loss_score = clamp01(1.0 - packet_loss)
    delay_score = clamp01(1.0 - delay / 50.0)

    p_stable = 0.45 * snr_score + 0.35 * loss_score + 0.20 * delay_score
    return round(clamp01(p_stable), 4)

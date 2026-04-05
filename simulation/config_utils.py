from __future__ import annotations

import config


def validate_config() -> None:
    if config.SOURCE_ID >= config.NUM_UAVS or config.DEST_ID >= config.NUM_UAVS:
        raise ValueError("SOURCE_ID hoặc DEST_ID đang vượt quá NUM_UAVS")

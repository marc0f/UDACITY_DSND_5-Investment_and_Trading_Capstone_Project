import os

# defaults symbols
SYMBOLS={
    "NRG": "NRG Energy Inc.",
    "VNO": "Vornado Realty Trust",
    "MGM": "MGM Resorts International",
    "ABC": "AmerisourceBergen Corp.",
    "ALGN": "Align Technology Inc.",
    "DXCM": "DexCom Inc.",
    "NVDA": "NVIDIA Corp.",
    "REGN": "Regeneron Pharmaceuticals Inc.",
    "^GSPC": "S&P 500"
}

# default data dir
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')


INTERVALS_TO_SECONDS={
    '1d': 24*60*60,
}

PREDICTION_HORIZONS = [1, 7, 14, 28]  # steps of prediction in base resolution, i.e. days
TEST_LEN_DAYS = 90  # days
TRAIN_LEN_DAYS = 2 * 265  # months

DASHBOARD_DATA_WINDOW_DAYS = 180


def interval_to_seconds(interval: str):

    seconds = INTERVALS_TO_SECONDS.get(interval)

    if seconds is None:
        raise ValueError("Missing reference value for given interval.")

    return seconds

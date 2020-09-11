import os

# defaults symbols
DEFAULT_SYMBOLS={
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


def interval_to_seconds(interval: str):

    seconds = INTERVALS_TO_SECONDS.get(interval)

    if seconds is None:
        raise ValueError("Missing reference value for given interval.")

    return seconds

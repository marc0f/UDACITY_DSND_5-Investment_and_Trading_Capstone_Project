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

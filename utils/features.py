

def sma(data, periods=10):
    return data.rolling(window=periods, min_periods=periods).mean()


def ema(data, periods=10):
    return data.ewm(span=periods, min_periods=periods, adjust=True).mean()


def vwap(price_data, volume_data, periods=10):
    volume_ma = volume_data.rolling(window=periods, min_periods=periods).mean()
    price_volume_ma = (price_data * volume_data).rolling(window=periods, min_periods=periods).mean()
    return price_volume_ma / volume_ma

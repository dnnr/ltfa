import colorsys

class LtfaError(Exception):
    """Error: {}"""


class ConfigError(Exception):
    """Configuration error: {}"""


def formatifnonempty(fmtstr: str, value: str) -> str:
    """
    Helper for formatting optional strings
    """
    if value != '':
        return fmtstr.format(value)
    else:
        return ''

def daycount_tostring(days: int) -> str:
    if days > 360:
        #  years = round(days / 365, ndigits=2)
        years = days / 365
        return "{:.2g} year{}".format(years, 's' if years > 1 else '')
    else:
        return "{} days".format(days)

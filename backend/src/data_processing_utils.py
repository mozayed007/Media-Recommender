import numpy as np


def process_date(x):
    """Pad incomplete date strings with missing month/day as '01'."""
    if isinstance(x, str):
        parts = x.split('-')
        if len(parts) < 3:
            if len(parts) == 1:
                return f"{x}-01-01"
            elif len(parts) == 2:
                return f"{x}-01"
        else:
            return x
    else:
        return np.nan

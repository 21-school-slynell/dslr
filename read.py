import pandas as pd
from libs import snackbar


def read_csv(name_file):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(name_file)
    except OSError:
        snackbar("Can\'t open file ({name})".format(name=name_file), 'error')
    except (ValueError, IndexError):
        snackbar("Not valid file ({name})".format(name=name_file), 'error')
    except Exception:
        snackbar("Unknown error", 'info')
    return df

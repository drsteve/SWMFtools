from functools import partial
import dateutil.parser as dup
import spacepy.datamodel as dm


def read_list(fname='richardson_cane_ICME_list.txt'):
    """Read Richardson-Cane ICME list from file

    Parameters
    ----------
    fname : str
        filename of Richardson-Cane ICME list

    Example
    -------
    >>> import readRC
    >>> data = readRC.read_list()
    """
    tfunc = partial(dup.parse, ignoretz=True)
    convdict = {'Epoch': tfunc,
                'ICME_start': tfunc,
                'ICME_end': tfunc,
                'Shock': bool,
                }
    data = dm.readJSONheadedASCII(fname, convert=convdict)
    return data


def get_event(rclist, index=0):
    """Get data for a given event number

    Parameters
    ----------
    rclist : dict-like
        SpaceData object returned by read_list
    index : int
        Integer index for event

    Example
    -------
    >>> import readRC
    >>> data = readRC.read_list()
    >>> readRC.get_event(data, 5)
    {'B_avg': 10.0,
     'Epoch': datetime.datetime(1996, 12, 23, 16, 0),
     'ICME_end': datetime.datetime(1996, 12, 25, 11, 0),
     'ICME_start': datetime.datetime(1996, 12, 23, 17, 0),
     'Shock': True,
     'V_avg': 360.0,
     'V_max': 420.0,
     'deltaV': 20.0}
    """
    evdict = {k: v[3] for k, v in rclist.items()}
    return evdict
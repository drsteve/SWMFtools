import requests
import re
import datetime as dt
from functools import partial
import numpy as np
import spacepy.datamodel as dm
from bs4 import BeautifulSoup

# first grab the HTML from the website
rc_url = 'http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm'

# If the proxies are set correctly in the environment, then
# the "get" request should just work. If having problems,
# set the proxy explicitly.
# proxies = {'http': "http://proxy.goes.here:port"
#            'https': "http://proxy.goes.here:port"
#            }
# html = requests.get(rc_url, proxies=proxies)
html = requests.get(rc_url)

# make a parser with BS4 and get header and data rows
soup = BeautifulSoup(html.text, 'html.parser')
# header is first element
table_header = soup.find("tr")
# each data row has a date in the first cell
data = [[cell.get_text(strip=True) for cell in row.find_all('td')]
        for row in soup.find_all('tr') if row.find("td", string=re.compile(r'\d{4}/\d'))]


# Now clean each row to extract dates/times and properties. Replace all '...' with NaN
def parse_date(instr):
    """input format 'YYYY/MM/DD HHMM'
    """
    tstr = instr.split("(")[0]
    time = dt.datetime.strptime(tstr, '%Y/%m/%d %H%M')
    return time


def dvint(instr, value=True):
    """get delta-V, or shock marker"""
    if 'S' in instr:
        shock = 1
    else:
        shock = 0
    vstr = instr.split()[0]
    try:
        val = np.int(vstr)
    except ValueError:
        val = np.nan
    if value:
        return val
    else:
        return shock


def parse_row(jdx, inrow, datadict):
    """change formatting/type of each row element
    """
    dvshock = partial(dvint, value=False)
    fields = [('Epoch', 0, parse_date),
              ('ICME_start', 1, parse_date),
              ('ICME_end', 2, parse_date),
              ('deltaV', 10, dvint), ('Shock', 10, dvshock),
              ('V_avg', 11, np.float),
              ('V_max', 12, np.float),
              ('B_avg', 13, np.float),
              # 'MC', 'Dst'
              ]
    for finame, idx, func in fields:
        datadict[finame][jdx] = func(inrow[idx])


# Organize output for writing to JSON-headed ASCII
nelem = len(data)
hcells = table_header.find_all('td')
outdata = dm.SpaceData(attrs={'DESCRIPTION': soup.find('title').get_text().strip(),
                              'SOURCE': rc_url,
                              'CREATION_DATE': dt.datetime.now().isoformat()})
# All the time values
outdata['Epoch'] = dm.dmfilled(nelem, fillval=np.nan, dtype=object,
                               attrs={'DESCRIPTION': hcells[0].get_text(strip=True)})
outdata['ICME_start'] = dm.dmfilled(nelem, fillval=np.nan, dtype=object,
                                    attrs={'DESCRIPTION': hcells[1].get_text(strip=True)})
outdata['ICME_end'] = dm.dmfilled(nelem, fillval=np.nan, dtype=object,
                                  attrs={'DESCRIPTION': hcells[1].get_text(strip=True)})
# And now the other variables we're interested in

# outdata['Comp_start'] = dm.dmfilled(nelem, fillval=np.nan, dtype=object)
# outdata['Comp_end'] = dm.dmfilled(nelem, fillval=np.nan, dtype=object)
# # Offset (hours) from Lepping- or Huttunen-reported times
# outdata['MC_start_offset'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.int)
# outdata['MC_end_offset'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.int)
# # Bidirectional streaming electrons
# outdata['BDE'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.int)
# # Bidirectional Ion Flows
# outdata['BIF'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.int)

# ICME characteristics
outdata['deltaV'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.float,
                                attrs={'DESCRIPTION': 'Increase in V at upstream disturbance',
                                       'UNITS': 'km/s'})
outdata['Shock'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.int,
                               attrs={'DESCRIPTION': 'Fast forward shock reported? 1 is True, 0 is False'})
outdata['V_avg'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.float,
                               attrs={'DESCRIPTION': 'Mean ICME speed',
                                      'UNITS': 'km/s'})
outdata['V_max'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.float,
                               attrs={'DESCRIPTION': 'Max solar wind speed during ICME',
                                      'UNITS': 'km/s'})
outdata['B_avg'] = dm.dmfilled(nelem, fillval=np.nan, dtype=np.float,
                               attrs={'DESCRIPTION': 'Mean magnetic field strength in ICME',
                                      'UNITS': 'nT'})

# Parse each row and fill target arrays
badrow = []
for idx, row in enumerate(data):
    try:
        parse_row(idx, row, outdata)
    except ValueError:
        badrow.append(idx)

# remove bad rows
odkeys = outdata.keys()
for odk in odkeys:
    outdata[odk] = np.delete(outdata[odk], badrow)

# Write to ASCII
varorder = ['Epoch', 'ICME_start', 'ICME_end', 'deltaV', 'Shock', 'V_avg',
            'V_max', 'B_avg']
outdata.toJSONheadedASCII('richardson_cane_ICME_list.txt', order=varorder)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[0:2, :], projection=ccrs.PlateCarree())
ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
ax1.gridlines(color='lightgrey', linestyle='-', draw_labels=True)

ax2 = fig.add_subplot(gs[2, :])
ax2.plot([1, 2], [3, 4])
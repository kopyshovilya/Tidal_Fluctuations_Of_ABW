import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from shapely.geometry import LineString
import scipy as sp
import scipy.interpolate
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.interpolate import interp1d
import sys
from scipy import interpolate
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seawater as sw
import scipy as sp
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator


dir = "F:\\Science\\Antarctic\\Profile_depth_4_moorages.txt"
dfB = pd.read_table(dir, sep=',')
zer = [float(dfB["Lat"][0]), float(dfB["Lon"][0])]
bot = np.zeros(len(dfB["Lon"]))
zer = [float(dfB["Lat"][0]), float(dfB["Lon"][0])]
for i in range(len(dfB["Lon"])):
    point = [float(dfB["Lat"][i]), float(dfB["Lon"][i])]
    bot[i] = sw.dist([zer[0], point[0]], [zer[1], point[1]])[0]



dirData = "F:\\Science\\Antarctic\\Data\\ES033_ctd.tab"
df = pd.read_table(dirData, sep='\t', skiprows = 187)
nameSt = ["ES033_ctd_094", "ES033_ctd_095", "ES033_ctd_096",
          "ES033_ctd_097", "ES033_ctd_098", "ES033_ctd_099",
          "ES033_ctd_100", "ES033_ctd_101", "ES033_ctd_055",
          "ES033_ctd_056", "ES033_ctd_057", "ES033_ctd_041", "ES033_ctd_058"]

# nameSt = ["ES033_ctd_094", "ES033_ctd_095"]
d = np.array([])
dep = np.array([])
char = np.array([])
bottomDist = np.array([])

SA = 100

for st_i in nameSt:
    dfSt = df[df["Event"] == st_i]
    lon = float(dfSt["Longitude"][:1])
    lat = float(dfSt["Latitude"][:1])
    x = sw.dist([zer[0], lat], [zer[1], lon])[0]
    if st_i == "ES033_ctd_041":
        x *= -1
    Sal = dfSt["Sal"]
    Sal_ = np.convolve(Sal, np.ones(SA) / SA, mode='same')[(SA//2):len(Sal)-(SA//2)]
    Temp = dfSt["Temp [°C]"]
    Temp_ = np.convolve(Temp, np.ones(SA) / SA, mode='same')[(SA//2):len(Temp)-(SA//2)]
    Dep = dfSt["Depth water [m]"]
    Dep_ = np.array(dfSt["Depth water [m]"][(SA//2):len(Temp)-(SA//2)])
    Press = dfSt["Press [dbar]"]
    Press_ = dfSt["Press [dbar]"][(SA//2):len(Temp)-(SA//2)]
    theta = sw.pden(Sal, Temp, Press)
    theta_ = sw.pden(Sal_, Temp_, Press_)
    ro = sw.dens(Sal, Temp, Press)
    x = x * np.ones(Dep_.shape[0]+1)

    N2 = np.abs(9.81/np.mean(theta) * np.gradient(theta, Dep))
    func_av = np.sin(np.linspace(0, np.pi, SA))
    func_av /= np.sum(func_av)
    N2_1 = np.convolve(N2, func_av, mode='same')[(SA//2):-(SA//2)]
    N2 = np.convolve(N2, np.ones(SA) / SA, mode='same')[(SA//2):-(SA//2)]

    Dep_  = np.append(Dep_ , Dep_[-1]+150)
    N2_1 = np.append(N2_1, N2_1[-1])
    d = np.append(d, x)


    dep  = np.append(dep , Dep_)
    # char = np.append(char, theta_)
    char = np.append(char, N2_1)



print(d.shape)
print(dep.shape)
print(char.shape)

fig, ax = plt.subplots(1, 1, figsize = (16, 7))

x = np.arange(0, max(d), 1)
y = np.arange(-10, max(dep), 1)
X, Y = np.meshgrid(x, y)
points = (np.array([d, dep])).T

interpolator = LinearNDInterpolator(list(zip(d, dep)), char)
Z = interpolator(X, Y)


Z[Y>50] = sp.ndimage.gaussian_filter(Z[Y>50], 0.5)

# plt.pcolormesh(x, y, Z-1000, vmin = 27.35, vmax = 27.85, zorder = -10, cmap = "PRGn", alpha = 0.5)
plt.contourf(x, y, np.log10(Z), vmin = -8, vmax = -4.5, levels = np.arange(-8, -4.4, 0.1), zorder = -10, cmap = "PRGn", alpha = 0.7)

clb = plt.colorbar()
# clb.set_label("Anomaly of potential density, kg m$^{-1}$", fontsize = 18, labelpad=-115)
clb.set_label("log$_{10}$(N$^2$ / s$^{-2}$)", fontsize = 18, labelpad=-115)
clb.set_ticks(np.arange(-8, -4.6, 0.5))
clb.ax.tick_params(labelsize = 18)

colorGoldBuet = '#feec62'

plt.plot(bot, -dfB["Dep"], 'k')
plt.fill_between(bot, -dfB["Dep"], np.zeros(bot.shape[0]) + 3000, color=colorGoldBuet)

depM5 = np.array([1502, 1554, 1606, 1658, 1762, 1813, 1839, 1856, 1891, 1907])
distM5 = sw.dist([zer[0], -74.169200], [zer[1], -29.543300])[0]

depM4 = np.array([636, 737, 790, 868, 895, 947, 973, 1026, 1042])
distM4 = sw.dist([zer[0], -74.438000], [zer[1], -30.044000])[0]

depM3 = np.array([363, 417, 468, 520, 571, 597, 623, 648, 700, 715])
distM3 = sw.dist([zer[0], -74.510500], [zer[1], -30.165100])[0]

depM3_Super = np.array([303, 328, 353, 378, 403, 428, 453, 478, 503, 529, 554, 579, 604, 632, 657, 682, 707, 716])
distM3_Super = sw.dist([zer[0], -74.507042722027], [zer[1], -30.156112514064])[0]

depM6 = np.array([328, 354, 379, 404, 431, 456, 471, 496, 505])
distM6 = sw.dist([zer[0], -74.5403], [zer[1], -30.1829])[0]

plt.plot(distM5 * np.ones(len(depM5)), depM5, 'k-')
plt.plot(distM4 * np.ones(len(depM4)), depM4, 'k-')
plt.plot(distM3 * np.ones(len(depM3)), depM3, 'k-')

colorRedBuetiful   = '#ee454a'
colorADCP = colorRedBuetiful

# plt.plot(distM6 * np.ones(len(depM6)), depM6, 'ro', ms = 4)
plt.plot(distM5 * np.ones(len(depM5)), depM5, 'ko', ms = 4)
plt.plot(distM4 * np.ones(len(depM4)), depM4, 'ko', ms = 4)
plt.plot(distM3 * np.ones(len(depM3)), depM3, 'ko', ms = 4)
# plt.plot(distM3_Super * np.ones(len(depM3_Super)), depM3_Super, 'ro', ms = 4)

ax.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("Depth, m", fontsize = 18)
plt.xlabel("Distance, km", fontsize = 18)

plt.xlim([0, 55])
plt.ylim([2000, 0])
plt.tight_layout()
         
# plt.savefig("F:\\Science\\Antarctic\\moorings_structrure\\Moorings_location_ES033_N2.png", dpi = 800, transparent=1)

plt.show()
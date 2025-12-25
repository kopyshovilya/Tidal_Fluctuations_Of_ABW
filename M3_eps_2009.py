import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from read_hyd_data import *
from elevation_N2 import *

import seawater as sw

dir = "data/M3_UV.tab"
df = pd.read_table(dir, sep='\t', skiprows = 20)
LAT = -74.510500
phi = -70 / 360 * 2 * np.pi
SA4T = 12*2
LA4T = 2

Vel1 = df[df["Gear ID"] == 1]
U1 = np.array(Vel1["Cur vel U [cm/s]"])
V1 = np.array(Vel1["Cur vel V [cm/s]"])

ADCP2 = df[df["Gear ID"] == 2]
U2 = np.array(ADCP2["Cur vel U [cm/s]"])
V2 = np.array(ADCP2["Cur vel V [cm/s]"])
depSensors2 = np.array([602, 598, 594, 590, 586, 582, 578, 574, 570, 566, 562, 558, 554, 550, 546, 542, 538, 534, 530, 526])

U2 = np.reshape(U2, (U2.shape[0]//depSensors2.shape[0], depSensors2.shape[0]))
V2 = np.reshape(V2, (V2.shape[0]//depSensors2.shape[0], depSensors2.shape[0]))

ADCP3 = df[df["Gear ID"] == 3]
U3 = np.array(ADCP3["Cur vel U [cm/s]"])
V3 = np.array(ADCP3["Cur vel V [cm/s]"])
depSensors3 = np.array([415, 410, 405, 400, 395, 390, 385, 380, 375, 370, 
                       365, 360, 355, 350, 310, 305, 300, 295, 290, 285, 
                       280, 275, 270, 265, 260, 255, 250, 245, 240, 235, 
                       230, 225, 220])

U3 = np.reshape(U3, (U3.shape[0]//depSensors3.shape[0], depSensors3.shape[0]))
V3 = np.reshape(V3, (V3.shape[0]//depSensors3.shape[0], depSensors3.shape[0]))

Time1 = np.array(Vel1["Date/Time"].unique(), dtype = 'datetime64')
Time2 = np.array(ADCP2["Date/Time"].unique(), dtype = 'datetime64')
Time3 = np.array(ADCP3["Date/Time"].unique(), dtype = 'datetime64')

Time2_, depSensors_ = np.meshgrid(Time2, depSensors2)

rotateSC = lambda U, V, angle: (U * np.sin(angle) + V * np.cos(angle), -(U * np.cos(angle) - V * np.sin(angle))) # return along and cross projections
U1_along, U1_cross =  rotateSC(U1, V1, phi)
U2_along, U2_cross =  rotateSC(U2, V2, phi)
U3_along, U3_cross =  rotateSC(U3, V3, phi)

fig, ax = plt.subplot_mosaic("AA;BB;CD;EF", figsize=(12, 8))

idxStart = 10
TimeLen = 25500

full_output = True

coreLoc = np.zeros([TimeLen])
coreSpeed = np.zeros([TimeLen])

dir = "F:\\Science\\Frey-moorings\\AntPen_moorings_from-Rinat2\\las_estaciones_de_buqueías\\Weddell_estaciones\\M3_TSP.tab"
data_processor = OceanographicData(dir)
ts = data_processor.get_parameter(['temperature', 'salinity'], instrument_index=[1, 2, 5])
ts['den'] = sw.pden(ts['salinity'], ts['temperature'], ts['depth'])

A = np.vstack([ts['temperature'], np.ones(len(ts['temperature']))]).T
k, b = np.linalg.lstsq(A, ts['den'], rcond=None)[0]
print(k, b)

temp_sensors = data_processor.get_parameter(['temperature'], instrument_index=[1, 2, 3, 4, 5])
temp_sensors['den'] = k * temp_sensors['temperature'] + b
depths = temp_sensors.depth.unique()
new_depths = np.convolve(depths, [0.5, 0.5], "valid")

dfM3 = pd.read_table(dir, sep='\t', skiprows = 21)
Sen1 = np.array(dfM3[dfM3["Depth water [m]"] == depths[1]])
Sen4 = np.array(dfM3[dfM3["Depth water [m]"] == depths[4]])
p1 = sw.dpth(Sen1[:, 3], 74.5)
p2 = sw.dpth(Sen4[:, 3], 74.5)

p1_mean = sw.dpth(np.min(Sen1[:, 3]), 74.5)
p2_mean = sw.dpth(np.min(Sen4[:, 3]), 74.5)

cos_angle = ( (p1 - p2) / (p1_mean-p2_mean)) # angle incl of mooring

xshape = int( (temp_sensors[temp_sensors['depth']  == depths[0]]).shape[0] )
n2 = np.empty([xshape, 4])
rho_mean = np.empty([xshape, 4])
gradRho = np.empty([xshape, 4])

for i in range(4):

    rho1 = np.array((temp_sensors[temp_sensors['depth']  == depths[i]])['den'])
    rho2 = np.array((temp_sensors[temp_sensors['depth']  == depths[i+1]])['den'])

    rho1 = np.convolve(rho1, np.ones(LA4T) / LA4T, mode='same')
    rho2 = np.convolve(rho2, np.ones(LA4T) / LA4T, mode='same')

    rho1_av = np.convolve(rho1, np.ones(LA4T) / LA4T, mode='same')
    rho2_av = np.convolve(rho2, np.ones(LA4T) / LA4T, mode='same')

    rho_mean[:, i] = (rho1/2 + rho2/2)
    rho_mean_av = (rho1_av/2 + rho2_av/2)
    n2[:, i] = np.abs(9.81 / rho_mean[:, i] * (rho2_av-rho1_av) / ( (depths[i] - depths[i+1])*(cos_angle) ))
    print("Mean BVF: ", np.mean(np.sqrt(n2[:, i])))
    gradRho[:, i] = (depths[i] - depths[i+1])*(cos_angle) / (rho2_av-rho1_av+1e-10) # d(z)/d(rho)
    # gradRho[:, i] = (depths[i] - depths[i+1]) / (rho2-rho1 + 0.000001) # d(z)/d(rho)
n2[n2 < 7.5e-7] = np.nan

depSensors = np.array([715,648,623,597,571,520,468,417,365])

TimeTSP = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[1]]['Date/Time'], dtype = 'datetime64')
########################### calc eps #################################

eps_Arr = np.empty([xshape, 4])
K_z_Arr = np.empty([xshape, 4])

for i in range(4):
    Rho1SA = np.convolve(rho_mean[:, i], np.ones(SA4T) / SA4T, mode='same')
    Rho1LA = rho_mean[:, i]

    eps_Arr[:, i] = 0.64 * (gradRho[:, i] * (Rho1SA - Rho1LA))**2 * (n2[:, i])**(3/2)
    K_z_Arr[:, i] = 0.128 * (gradRho[:, i] * (Rho1SA - Rho1LA))**2 * (n2[:, i])**(1/2)
    print(gradRho[:, i])

epslim = np.array([1e-10, 1e-4])
ax['B'].set_yscale('log')
ax['B'].set_ylim([1e-10, 1e-4])
ax['E'].set_yscale('log')
ax['E'].set_ylim([1e-10, 1e-4])
ax['F'].set_yscale('log')
ax['F'].set_ylim([1e-10, 1e-4])


col = ['magenta', 'k', 'r', 'b']
col = ['r', 'g', 'b', 'k']

for i in range(0, 4):
    eps = eps_Arr[:, i]
    SA4T = 6
    eps_ = np.convolve(eps, np.ones(SA4T) / SA4T, mode='same')
    eps_Arr[:, i] = eps_
max_eps = np.nanmax(eps_Arr, axis=1)
min_eps = np.nanmin(eps_Arr, axis=1)
print("Max eps ", np.nanmean(max_eps[100:-100]))
print("Min eps ", np.nanmean(min_eps[100:-100]))
min_eps = np.nanmin(eps_Arr, axis=1)
# A B C D E F - 10 11 12 13 14 15
# A2 C9 72
colorOrangeBuetiful = '#ff885A'
colorBlueBuetiful   = 'darkgreen'

ax['B'].plot(TimeTSP, min_eps, color = colorOrangeBuetiful, lw = 0.5)
ax['E'].plot(TimeTSP, min_eps, color = colorOrangeBuetiful, lw = 0.5)
ax['F'].plot(TimeTSP, min_eps, color = colorOrangeBuetiful, lw = 0.5)

ax['B'].plot(TimeTSP, max_eps, color = colorBlueBuetiful, lw = 0.5)
ax['E'].plot(TimeTSP, max_eps, color = colorBlueBuetiful, lw = 0.5)
ax['F'].plot(TimeTSP, max_eps, color = colorBlueBuetiful, lw = 0.5)

#######################################################################

LA = 1 * 60 // 5
coreLoc = np.convolve(coreLoc, np.ones(LA) / LA, mode='same')


ax['A'].plot(Time2, np.nanmean(U2_along, axis=1), 'r', label = 'Along speed', lw = 0.5)
ax['A'].plot(Time2, np.nanmean(U2_cross, axis=1), 'b', label = 'Cross speed', lw = 0.5)
ax['A'].fill_between(Time2, np.nanmean(U2_along, axis=1), np.zeros(U2_along.shape[0]), color="r", alpha=0.3)
ax['A'].fill_between(Time2, np.nanmean(U2_cross, axis=1), np.zeros(U2_cross.shape[0]), color="b", alpha=0.3)
ax['A'].legend(fontsize = 10, loc = 'lower left')
ax['C'].plot(Time2, np.nanmean(U2_along, axis=1), 'r', label = 'Along speed', lw = 0.5)
ax['C'].plot(Time2, np.nanmean(U2_cross, axis=1), 'b', label = 'Cross speed', lw = 0.5)
ax['C'].fill_between(Time2, np.nanmean(U2_along, axis=1), np.zeros(U2_along.shape[0]), color="r", alpha=0.3)
ax['C'].fill_between(Time2, np.nanmean(U2_cross, axis=1), np.zeros(U2_cross.shape[0]), color="b", alpha=0.3)
ax['D'].plot(Time2, np.nanmean(U2_along, axis=1), 'r', label = 'Along speed', lw = 0.5)
ax['D'].plot(Time2, np.nanmean(U2_cross, axis=1), 'b', label = 'Cross speed', lw = 0.5)
ax['D'].fill_between(Time2, np.nanmean(U2_along, axis=1), np.zeros(U2_along.shape[0]), color="r", alpha=0.3)
ax['D'].fill_between(Time2, np.nanmean(U2_cross, axis=1), np.zeros(U2_cross.shape[0]), color="b", alpha=0.3)


tlimALL = np.array(['2009-02-14T00:00', '2010-02-09T00:00'], dtype = 'datetime64')
tlim1 = np.array(['2009-05-05T00:00', '2009-05-08T12:00'], dtype = 'datetime64')
tlim1 = np.array(['2009-03-30T00:00', '2009-04-02T12:00'], dtype = 'datetime64')
tlim2 = np.array(np.array(['2009-09-15T00:00', '2009-09-19T00:00'], dtype = 'datetime64'))
ylim = np.array([-60, 60])

rect11 = patches.Rectangle((tlim1[0], ylim[0]), (tlim1[1] - tlim1[0]), (ylim[1] - ylim[0]), linewidth=0, edgecolor='r', facecolor='lime', alpha = 0.4)
rect12 = patches.Rectangle((tlim2[0], ylim[0]), (tlim2[1] - tlim2[0]), (ylim[1] - ylim[0]), linewidth=0, edgecolor='r', facecolor='lime', alpha = 0.4)
rect21 = patches.Rectangle((tlim1[0], epslim[0]), (tlim1[1] - tlim1[0]), (epslim[1] - epslim[0]), linewidth=0, edgecolor='r', facecolor='lime', alpha = 0.4)
rect22 = patches.Rectangle((tlim2[0], epslim[0]), (tlim2[1] - tlim2[0]), (epslim[1] - epslim[0]), linewidth=0, edgecolor='r', facecolor='lime', alpha = 0.4)
ax['A'].add_patch(rect11)
ax['A'].add_patch(rect12)
ax['B'].add_patch(rect21)
ax['B'].add_patch(rect22)

ax['A'].set_xlim(tlimALL)
ax['B'].set_xlim(tlimALL)
ax['C'].set_xlim(tlim1)
ax['E'].set_xlim(tlim1)
ax['D'].set_xlim(tlim2)
ax['F'].set_xlim(tlim2)

ax['A'].set_ylim(ylim)
ax['C'].set_ylim([-30, 30])
ax['D'].set_ylim([-30, 30])

ax['A'].set_ylabel("Velocity, cm s$^{-1}$", fontsize = 12)
ax['B'].set_ylabel("$<\epsilon>, m^2 s^{-3}$", fontsize = 12)
ax['C'].set_ylabel("Velocity, cm s$^{-1}$", fontsize = 12)
ax['E'].set_ylabel("$<\epsilon>, m^2 s^{-3}$", fontsize = 12)
ax['E'].set_xlabel("Time", fontsize = 12)
ax['F'].set_xlabel("Time", fontsize = 12)

ax['E'].fill_between(TimeTSP, max_eps, np.ones(max_eps.shape[0]) * np.nanmean(max_eps[100:-100]), color=colorBlueBuetiful, alpha=0.3)
ax['E'].fill_between(TimeTSP, min_eps, np.ones(min_eps.shape[0]) * np.nanmean(min_eps[100:-100]), color=colorOrangeBuetiful, alpha=0.3)

ax['F'].fill_between(TimeTSP, max_eps, np.ones(max_eps.shape[0]) * np.nanmean(max_eps[100:-100]), color=colorBlueBuetiful, alpha=0.3)
ax['F'].fill_between(TimeTSP, min_eps, np.ones(min_eps.shape[0]) * np.nanmean(min_eps[100:-100]), color=colorOrangeBuetiful, alpha=0.3)

# ax['E'].plot(TimeTSP, min_eps, color = colorOrangeBuetiful, lw = 0.5)

ax['A'].get_xaxis().set_visible(False)
ax['C'].get_xaxis().set_visible(False)
ax['D'].get_xaxis().set_visible(False)

ax['E'].xaxis.get_ticklocs(minor=True)
ax['E'].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax['E'].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax['E'].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax['E'].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))

ax['F'].xaxis.get_ticklocs(minor=True)
ax['F'].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax['F'].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax['F'].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax['F'].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))


plt.subplots_adjust(left = 0.07, bottom = 0.06, right = 0.97, top = 0.98, wspace=0.13, hspace=0.14)

# plt.savefig('eps_months4.png', dpi = 600)
print(2*np.pi/sw.f(74.510500) / 3600)
print(2*np.pi/sw.f(74.438000) / 3600)
print(2*np.pi/sw.f(74.169200) / 3600)
plt.show()



import numpy as np
import pandas as pd
import scipy.integrate
import matplotlib.pyplot as plt
from read_hyd_data import *
from elevation_N2 import *

import seawater as sw
from scipy.signal import welch

def spectrPlot(data, dT, N, axPlot, col, labelName = None, windowAver = 10, timeWin = 200, alpha = 1):
    data = np.array(data)

    freq, sp_sum = welch(data, fs=(1/dT), nperseg=(timeWin/dT))
    sp_sum = (np.abs(sp_sum))
    axPlot.plot(freq[2:-1], np.abs(sp_sum[2:-1]), c=col, label = labelName, alpha = alpha)
    return freq, sp_sum

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

LAT = -74.438000
phi = -70 / 360 * 2 * np.pi
SA4T = 12*1
LA4T = 12*1

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

idxStart = 10
TimeLen = 25500

dir = "data/M5_TSP.tab"
data_processor = OceanographicData(dir)
ts = data_processor.get_parameter(['temperature', 'salinity'], instrument_index=[1, 2])
ts['den'] = sw.pden(ts['salinity'], ts['temperature'], ts['depth'])

A = np.vstack([ts['temperature'], np.ones(len(ts['temperature']))]).T
k, b = np.linalg.lstsq(A, ts['den'], rcond=None)[0]

countSen = 8

temp_sensors = data_processor.get_parameter(['temperature'], instrument_index=[2, 3, 4, 5, 6, 7, 8, 9, 1])
temp_sensors['den'] = k * temp_sensors['temperature'] + b
depths = temp_sensors.depth.unique()
new_depths = np.convolve(depths, [0.5, 0.5], "valid")

print(new_depths)
print(depths)

dfM5 = pd.read_table(dir, sep='\t', skiprows = 21)
Sen1 = np.array(dfM5[dfM5["Depth water [m]"] == depths[2]])
Sen4 = np.array(dfM5[dfM5["Depth water [m]"] == depths[5]])
p1 = sw.dpth(Sen1[:, 3], LAT)
p2 = sw.dpth(Sen4[:, 3], LAT)
print(p1)
print(p2)

p1_mean = sw.dpth(np.min(Sen1[:, 3]), LAT)
p2_mean = sw.dpth(np.min(Sen4[:, 3]), LAT)

cos_angle = ( (p1 - p2) / (p1_mean-p2_mean)) # angle incl of mooring

xshape = int( (temp_sensors[temp_sensors['depth']  == depths[0]]).shape[0] )
n2 = np.empty([xshape, countSen])
n2_fit = np.empty([xshape, countSen])
ksi = np.empty([xshape, countSen])
n2_mean = np.empty([xshape, countSen])
rho_mean = np.empty([xshape, countSen])
gradRho = np.empty([xshape, countSen])

for i in range(countSen):
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
dT = 5*60
for i in range(countSen):
    n2_fit[:, i] = lowpass(n2[:, i], 1/(48*3600), 1/dT)
n2_mean = np.mean(n2, axis=1)
for i in range(countSen):
    ksi[:, i] = (n2[:, i] - n2_fit[:, i]) / n2_mean
# depSensors = np.array([715,648,623,597,571,520,468,417,365])
TimeTSP = np.array(dfM5[dfM5["Depth water [m]"] == depths[1]]['Date/Time'], dtype = 'datetime64')

########################### calc elevation #################################

elev_Arr = np.empty([xshape, countSen])
spe_12 = np.empty([countSen])
spe_24 = np.empty([countSen])

for i in range(countSen):

    Rho1SA = np.convolve(rho_mean[:, i], np.ones(SA4T) / SA4T, mode='same')
    # Rho1LA = np.convolve(rho_mean[:, i], np.ones(LA4T) / LA4T, mode='same')
    Rho1LA = rho_mean[:, i]

    elev_Arr[:, i] = (gradRho[:, i] * (Rho1SA - Rho1LA))

colorOrangeBuetiful = '#ff885A'
colorGreenBuetiful  = '#a2c772'
colorBlueBuetiful   = '#6c76b7'
colorRedBuetiful   = '#ee454a'
colorPurple = '#a674c0'
colorPink = '#e786cb'
col = [colorBlueBuetiful, colorOrangeBuetiful, colorGreenBuetiful, colorRedBuetiful, colorPurple, colorPink, 'k', 'cyan']
# col = ['r', 'orange', 'yellow', 'green', 'blue', 'darkblue', 'purple']

x1_24 = 1/(23.5 * 3600)
x2_24 = 1/(24.5 * 3600)

x1_12 = 1/(11.5 * 3600)
x2_12 = 1/(12.5 * 3600)

print(x2_12-x1_12)
print(x2_24-x1_24)
for i in range(countSen):
    elev = elev_Arr[:, i]
    elev = ksi[:, i]
    
    freq, sp_sum = spectrPlot(elev[5:-5], dT = (5*60), N=len(elev)-10, axPlot=ax, col=col[i], timeWin = 24*30*2*3600, alpha = 0.25)
    winAvSp = 6
    sp_sum = np.convolve(sp_sum, np.ones(winAvSp) / winAvSp, mode='same')
    ax.plot(freq[winAvSp//2:-winAvSp//2], sp_sum[winAvSp//2:-winAvSp//2], c=col[i], lw = 2, label = str(new_depths[i]) + ' m')
    dnu = np.mean(np.diff(freq))
    sp_24 = sp_sum[(freq < x1_24) & (freq > x2_24)]
    freq_24 = freq[(freq < x1_24) & (freq > x2_24)]

    sp_12 = sp_sum[(freq < x1_12) & (freq > x2_12)]
    freq_12 = freq[(freq < x1_12) & (freq > x2_12)]

    spe_24[i] = scipy.integrate.trapezoid(sp_24, freq_24, dnu) / (x1_24-x2_24)
    spe_12[i] = scipy.integrate.trapezoid(sp_12, freq_12, dnu) / (x1_12-x2_12)
ax.set_xlabel('Frequency, s$^{-1}$')
ax.set_ylabel('$ξ_z$ spectrum, s$^{1}$')
T =    np.array([12, 24, 13, 31])
T =    np.array([12.42, 12, 12.66, 11.97, 25.82, 23.93, 24.07, 26.87, 327.86, 661.3, 40, 3])

f_ = sw.f(74.510500) / 2 / 3.1415 * 3600

ax.set_ylim([5e2, 4e7])
ax.set_xlim([1/(48*3600), 1/(2*3600)])

ax.set_yscale('log')
ax.set_xscale('log')

ax_time = ax.twiny()
ax_time.set_xscale('log')
ax_time.set_xlim([48, 2])
ax_time.set_ylim([1e0, 1e7])

locTick = [48, 24, 12, 6, 3, 2]
ax_time.set_xticks(locTick)
ax_time.set_xlabel('Period, hour')
ax_time.set_xticklabels(locTick)
ax_time.xaxis.get_ticklocs(minor=True)

ax_time.fill_betweenx([0, 1e10], 23, 25, color="k", alpha=0.3, zorder = -10)
ax_time.fill_betweenx([0, 1e10], 11.5, 12.5, color="k", alpha=0.3, zorder = -10)

plt.tight_layout()
plt.savefig("Spectrum_fig\\Ksi_M5_spectrum.png", dpi = 1000)

f, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.xlim([1e4, 0])
plt.ylabel("Depth, m")
plt.xlabel("SPE, s")
plt.plot(spe_12, new_depths, c='darkblue')
plt.ylim([np.max(new_depths), np.min(new_depths)])
plt.fill_betweenx(new_depths, 0, spe_12, color='darkblue', alpha = 0.3)
plt.tight_layout()
plt.savefig('Spectrum_fig/SPE_M5_12_profile.png', dpi = 500, transparent=1)

f, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.xlim([1e4, 0])
plt.ylabel("Depth, m")
plt.xlabel("SPE, s")
plt.plot(spe_24, new_depths, c='darkblue')
plt.ylim([np.max(new_depths), np.min(new_depths)])
plt.fill_betweenx(new_depths, 0, spe_24, color='darkblue')
plt.tight_layout()
plt.savefig('Spectrum_fig/SPE_M5_24_profile.png', dpi = 500, transparent=1)
plt.show()

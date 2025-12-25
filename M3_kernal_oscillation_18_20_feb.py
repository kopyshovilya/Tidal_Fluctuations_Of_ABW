import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import leastsq
import matplotlib.ticker as ticker
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


fig = plt.figure(figsize=(16, 8), constrained_layout=True)
widths = [1]
heights = [2, 1, 1.5]
gs = fig.add_gridspec(ncols=1, nrows=3, width_ratios=widths, height_ratios=heights)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

idxStart = 10
TimeLen = 1000

quadratic_func = lambda x, b0, b1, b2: b0 + b1*x + b2*x**2
qubic_func = lambda x, b0, b1, b2, b3: b0 + b1*x + b2*x**2 + b3*x**3
func2 = quadratic_func
func3 = qubic_func
residual_func2 = lambda p, x, y: y - func2(x, *p)
residual_func3 = lambda p, x, y: y - func3(x, *p)
full_output = True

stream = np.zeros([TimeLen, len(depSensors2)])
stream_parabolic = np.zeros([TimeLen, len(depSensors2)])
stream_3 = np.zeros([TimeLen, len(depSensors2)])

stream_cross_quadr = np.zeros([TimeLen, len(depSensors2)])
stream_along_quadr = np.zeros([TimeLen, len(depSensors2)])

coreLoc = np.zeros([TimeLen])
coreSpeed = np.zeros([TimeLen])

dir = "data/M3_TSP.tab"
dfM3 = pd.read_table(dir, sep='\t', skiprows = 21)
TimeM3 = np.array(dfM3[dfM3["Gear ID"] == 1]['Date/Time'], dtype = 'datetime64')
SA = 48 * 60 // 5
TimeM3 = TimeM3[(SA//2):-(SA//2)]
depSensors = np.array([715,648,623,597,571,520,468,417,365])



Sen1 = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[0]])
Sen4 = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[4]])
TimeTSP = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[1]]['Date/Time'], dtype = 'datetime64')
Ri = np.zeros([TimeLen, len(depSensors2)-1])
maxRi = np.zeros([TimeLen])
Ri_down = np.zeros([TimeLen])
N_2_ = np.zeros([TimeLen])

for i in tqdm(range(idxStart, idxStart + TimeLen)):

    U_along, U_cross = (U2_along[i], U2_cross[i])

    Cur = np.sqrt(U_along**2 + U_cross**2)
    Cur_init = Cur.copy()

    dataSen1 = Sen1[TimeTSP == Time2[i]]
    dataSen4 = Sen4[TimeTSP == Time2[i]]
    ro1 = sw.dens(dataSen1[0, 5], dataSen1[0, 4], sw.pres(715, LAT))
    ro4 = sw.dens(dataSen4[0, 5], dataSen4[0, 4], dataSen4[0, 3])
    dep1 = sw.dpth(dataSen1[0, 3], LAT)
    dep1 = 715
    dep4 = sw.dpth(dataSen4[0, 3], LAT)
    N_2 = 9.81 / (ro1/2 + ro4/2) * (ro4-ro1) / (dep4 - dep1)
    ################# eps ######################


    ################# Ri #######################
    Ri_i = N_2 / ((np.diff(U_along)/400)**2 + (np.diff(U_cross)/400)**2)
    Ri[i-idxStart] = Ri_i
    N_2_[i-idxStart] = N_2
    maxRi[i-idxStart] = np.max(Ri_i)
    dataU1_along = U1_along[(Time1 > Time2[i] - np.timedelta64(6,'m')) & (Time1 < Time2[i] + np.timedelta64(6,'m'))]
    dataU1_cross = U1_cross[(Time1 > Time2[i] - np.timedelta64(6,'m')) & (Time1 < Time2[i] + np.timedelta64(6,'m'))]

    if dataU1_along.shape[0] != 0:
        Ri_down[i-idxStart] = N_2 / (((dataU1_cross - U_cross[-1])/17400)**2 + ((dataU1_along - U_along[-1])/17400)**2)
    ############################################
    # interpolation to Nan pixel
    ############################################
    depSenWNanU_along = depSensors2[~np.isnan(U_along)]
    depSenWNanU_cross = depSensors2[~np.isnan(U_cross)]
    depSenWNanCur = depSensors2[~np.isnan(Cur)]

    U_along = U_along[~np.isnan(U_along)]
    U_cross = U_cross[~np.isnan(U_cross)]
    Cur = Cur[~np.isnan(Cur)]

    # newDep = np.array([depSensors2[0], depSensors2[10], depSensors2[-1]])
    p0_2 = [0, 0, 0]
    p0_3 = [0, 0, 0, 0]
    (popt, cov_x, infodict, mesg, ier) = leastsq(residual_func2, p0_2, args=(depSenWNanCur, Cur), full_output=full_output)
    Cur_parabolic = quadratic_func(depSensors2, popt[0], popt[1], popt[2])

    p0_2 = [0, 0, 0]
    (popt, cov_x, infodict, mesg, ier) = leastsq(residual_func2, p0_2, args=(depSenWNanU_along, U_along), full_output=full_output)
    U_along_parabolic = quadratic_func(depSensors2, popt[0], popt[1], popt[2])

    p0_2 = [0, 0, 0]
    (popt, cov_x, infodict, mesg, ier) = leastsq(residual_func2, p0_2, args=(depSenWNanU_cross, U_cross), full_output=full_output)
    U_cross_parabolic = quadratic_func(depSensors2, popt[0], popt[1], popt[2])

    stream_parabolic[i-idxStart] = Cur_parabolic / np.mean(Cur_parabolic)
    stream[i-idxStart] = Cur_init / np.mean(Cur_init)
    stream_cross_quadr[i-idxStart] = U_cross_parabolic # cross the shelf
    stream_along_quadr[i-idxStart] = U_along_parabolic # along the shelf
    coreLoc[i-idxStart] = depSensors2[Cur_parabolic == np.max(Cur_parabolic)]
    coreSpeed[i-idxStart] = np.max(np.abs(Cur))

########################### calc eps #################################

from read_hyd_data import *
from elevation_N2 import *
dir = "data/M3_TSP.tab"
data_processor = OceanographicData(dir)
ts = data_processor.get_parameter(['temperature', 'salinity'], instrument_index=[1, 2, 5])
ts['den'] = sw.pden(ts['salinity'], ts['temperature'], ts['depth'])

A = np.vstack([ts['temperature'], np.ones(len(ts['temperature']))]).T
k, b = np.linalg.lstsq(A, ts['den'], rcond=None)[0]

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
    gradRho[:, i] = (depths[i] - depths[i+1])*(cos_angle) / (rho2_av-rho1_av+1e-10) # d(z)/d(rho)
TimeTSP = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[1]]['Date/Time'], dtype = 'datetime64')
eps_Arr = np.empty([xshape, 4])
K_z_Arr = np.empty([xshape, 4])
col = ['darkgreen', 'k', 'r', 'orange']

for i in range(4):
    Rho1SA = np.convolve(rho_mean[:, i], np.ones(SA4T) / SA4T, mode='same')
    # Rho1LA = np.convolve(rho_mean[:, i], np.ones(LA4T) / LA4T, mode='same')
    Rho1LA = rho_mean[:, i]

    eps_Arr[:, i] = 0.64 * (gradRho[:, i] * (Rho1SA - Rho1LA))**2 * (n2[:, i])**(3/2)
    K_z_Arr[:, i] = 0.2 * eps_Arr[:, i] * (n2[:, i])**(-1)
colorOrangeBuetiful = '#ff885A'
colorBlueBuetiful   = 'darkgreen'
max_eps = np.nanmax(eps_Arr, axis=1)
min_eps = np.nanmin(eps_Arr, axis=1)
ax3.plot(TimeTSP, max_eps, c = colorOrangeBuetiful, label = '$\epsilon$')
ax3.plot(TimeTSP, min_eps, c = colorOrangeBuetiful, label = '$\epsilon$')
ax3.fill_between(TimeTSP, min_eps, max_eps, color=colorOrangeBuetiful, alpha=0.3)


ax3.set_yscale('log')
ax3.set_ylim([1e-11, 1e-5])
ax3.set_ylabel('$<\epsilon>, m^2 s^{-3}$', fontsize = 14)

#######################################################################

LA = 1 * 60 // 5
coreLoc = np.convolve(coreLoc, np.ones(LA) / LA, mode='same')
ax1.plot(Time2[idxStart: idxStart + TimeLen], -coreLoc, 'gold')

data = np.zeros(stream_parabolic.shape)
data[stream_parabolic > 1] = 1
data[stream_parabolic > 2] = 2
data[stream_parabolic > 3] = 3
Speed_cross = np.zeros(stream_parabolic.shape)
Speed_cross[stream_cross_quadr > 0] = 1
Speed_cross[stream_cross_quadr < 0] = -1

Speed_along = np.zeros(stream_parabolic.shape)
Speed_along[stream_along_quadr > 0.5] = 1
Speed_along[stream_along_quadr < 0.5] = -1

g1 = ax1.pcolormesh(Time2[idxStart: idxStart + TimeLen], -depSensors2, data.T)


Time2_grid, depGrid = np.meshgrid(Time2[idxStart: idxStart + TimeLen:6], -depSensors2[::3])

ax1.quiver(Time2_grid, depGrid,
             stream_along_quadr[::6,  ::3], 
             stream_cross_quadr[::6,  ::3], 
             color = 'r', zorder = 10, scale = 1000)

ax1.quiver(Time1[idxStart//3: idxStart//3 + TimeLen//3], 
             -700*np.ones(len(Time1[idxStart//3: idxStart//3 + TimeLen//3])),
             U1_along[idxStart//3: idxStart//3 + TimeLen//3], 
             U1_cross[idxStart//3: idxStart//3 + TimeLen//3], 
             color = 'k', width = 0.003, scale = 1000)

ax1.quiver(np.array(['2009-02-18T18:00'], dtype = 'datetime64'), -625, 20, 0, color = 'r', width = 0.003, scale=1000)


#################### Plot isotherms #################################

idTs = [1, 3, 5]

for idT in idTs:
    T = np.array(sw.eos80.ptmp(34.5, dfM3[dfM3["Depth water [m]"] == depSensors[idT]]['Temp [°C]'], depSensors[idT]))
    TSA = np.convolve(T, np.ones(SA4T) / SA4T, mode='same')
    TimeT = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[idT]]['Date/Time'], dtype = 'datetime64')
    ax1.plot(TimeT, -depSensors[idT] + T*50, 'lime')
idTs = [0, 3, 6]

    

########################################################################

ax2.plot(Time2, np.mean(U2_along, axis=1), 'r', label = 'Along speed')
ax2.plot(Time2, np.mean(U2_cross, axis=1), 'b', label = 'Cross speed')
ax2.fill_between(Time2, np.nanmean(U2_along, axis=1), np.zeros(U2_along.shape[0]), color="r", alpha=0.3)
ax2.fill_between(Time2, np.nanmean(U2_cross, axis=1), np.zeros(U2_cross.shape[0]), color="b", alpha=0.3)
ax2.legend(fontsize = 14)

ax1.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))
ax2.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))
ax3.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))

ax1.set_ylim([-725, -525])
ax2.set_ylim([-35, 35])

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

ax3.xaxis.get_ticklocs(minor=True)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax3.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))


ax1.set_ylabel("Depth, m", fontsize = 14)
ax2.set_ylabel("Velocity, cm s$^{-1}$", fontsize = 14)

# plt.tight_layout()
# plt.savefig('cascading_eps_new.png', dpi = 800)
plt.show()



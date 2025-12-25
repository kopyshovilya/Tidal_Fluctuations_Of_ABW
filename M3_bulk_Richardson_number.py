from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seawater as sw
from read_hyd_data import *
from elevation_N2 import *
import scipy.interpolate as interp

dir = "data/M3_UV.tab"
df = pd.read_table(dir, sep='\t', skiprows = 20)
LAT = -74.510500
phi = -70 / 360 * 2 * np.pi # rotate angle 
SA4T = 6
LA4T = 1

Vel1 = df[df["Gear ID"] == 1]
U1 = np.array(Vel1["Cur vel U [cm/s]"]) / 100 # in m/s
V1 = np.array(Vel1["Cur vel V [cm/s]"]) / 100 # in m/s

ADCP2 = df[df["Gear ID"] == 2]
U2 = np.array(ADCP2["Cur vel U [cm/s]"]) / 100 # in m/s
V2 = np.array(ADCP2["Cur vel V [cm/s]"]) / 100 # in m/s
depSensors2 = np.array([602, 598, 594, 590, 586, 582, 578, 574, 570, 566, 
                        562, 558, 554, 550, 546, 542, 538, 534, 530, 526])

U2 = np.reshape(U2, (U2.shape[0]//depSensors2.shape[0], depSensors2.shape[0]))
V2 = np.reshape(V2, (V2.shape[0]//depSensors2.shape[0], depSensors2.shape[0]))

ADCP3 = df[df["Gear ID"] == 3]
U3 = np.array(ADCP3["Cur vel U [cm/s]"]) / 100 # in m/s
V3 = np.array(ADCP3["Cur vel V [cm/s]"]) / 100 # in m/s
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

idxStart = 100
TimeLen = 2500

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

# Calculate density  from temperature

dir = "Data\\M3_TSP.tab"
data_processor = OceanographicData(dir) # Load data
ts = data_processor.get_parameter(['temperature', 'salinity'], instrument_index=[1, 2, 5]) # setup 1, 2, 5 instriments (with Temperature and conductivity measurments)
ts['den'] = sw.pden(ts['salinity'], ts['temperature'], ts['depth']) # calculate density

A = np.vstack([ts['temperature'], np.ones(len(ts['temperature']))]).T
k, b = np.linalg.lstsq(A, ts['den'], rcond=None)[0] # calculate linear approximation coefficients

temp_sensors = data_processor.get_parameter(['temperature'], instrument_index=[1, 2, 3, 4, 5, 8]) # setup 1, 2, 3, 4, 5, 8 instriments
temp_sensor_520 = data_processor.get_parameter(['temperature'], instrument_index=[8])

temp_sensors['den'] = k * temp_sensors['temperature'] + b # calculate "density"
temp_sensor_520['den'] = k * temp_sensors['temperature'] + b # calculate "density"



depths = temp_sensors.depth.unique() # depth of instruments

t1 = np.arange(0, len(temp_sensors['date_time'].unique())*5, 5)
t2 = np.arange(0, len(temp_sensor_520['date_time'])*20, 20)
Rho520 = interp.interp1d(t2, temp_sensor_520['den'])(t1)        # resample data 

Rho571 = np.array(temp_sensors[temp_sensors['depth'] == depths[4]]['den'])
Rho597 = np.array(temp_sensors[temp_sensors['depth'] == depths[3]]['den'])

Rho_lower = np.empty([temp_sensors[temp_sensors['depth'] == depths[0]]['den'].shape[0], 4])
depths_lower = depths[:4]
for i in range(4):
    Rho_lower[:, i] = temp_sensors[temp_sensors['depth'] == depths[i]]['den']

Win = 100
Rho520 = np.convolve(Rho520, np.ones(Win)/Win, mode = 'same')
Rho571 = np.convolve(Rho571, np.ones(Win)/Win, mode = 'same')
Rho597 = np.convolve(Rho597, np.ones(Win)/Win, mode = 'same')
for i in range(3):
    Rho_lower[:, i] = np.convolve(Rho_lower[:, i], np.ones(Win)/Win, mode = 'same')

dz = np.diff(depths)
N2_data = pd.DataFrame({'date_time': np.array([]),
                        'depth': np.array([]),
                        'N2': np.array([])})
xshape = int( (temp_sensors[temp_sensors['depth']  == depths[0]]).shape[0] )

dfM3 = pd.read_table(dir, sep='\t', skiprows = 21)
TimeM3 = np.array(dfM3[dfM3["Gear ID"] == 1]['Date/Time'], dtype = 'datetime64')
SA = 48 * 60 // 5
TimeM3 = TimeM3[(SA//2):-(SA//2)]
depSensors = np.array([715,648,623,597,571,520,468,417,365])

Sen5 = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[4]]) # TSD sensors with delta time 5 min
Sen9 = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[8]]) # TSD sensors with delta time 5 min
TimeTSP = np.array(dfM3[dfM3["Depth water [m]"] == depSensors[1]]['Date/Time'], dtype = 'datetime64')
Ri_bulk = np.zeros([TimeLen, 3])
maxRi = np.zeros([TimeLen])
Ri_down = np.zeros([TimeLen])
N_2_ = np.zeros([TimeLen])

# cycle each 20 min = 4 parts dt=5 min 



for i in tqdm(range(idxStart, idxStart + TimeLen)):
    U_along, U_cross = (U2_along[i//4], U2_cross[i//4])

    Cur = np.sqrt(U_along**2 + U_cross**2)
    Cur_init = Cur.copy()

    dep1 = 715

    ################ grad U ####################
    # print(Time1[i])
    Cur_bottom = np.sqrt(U1_along[i//12]**2 + U1_cross[i//12]**2)

    gradU = np.ones([3])
    gradU[0] = np.nanmean(np.gradient(Cur_init[9:], np.arange(566, 522, -4)))       # 526 - 566 m 
    gradU[1] = np.nanmean(np.gradient(Cur_init[1:9], np.arange(598, 566, -4)))      # 570 - 598 m 
    gradU[2] = (Cur_init[0] - Cur_bottom) / (602 - 700)                          # 602 - 700 m 

    ################ grad Rho ####################

    gradRho = np.ones([3])
    gradRho[0] = (np.mean(Rho520[i: i+4], axis = 0) - np.mean(Rho571[i: i+4], axis = 0)) / (571 - 520)          # 526 - 566 m 
    gradRho[1] = (np.mean(Rho571[i: i+4], axis = 0) - np.mean(Rho597[i: i+4], axis = 0)) / (597 - 571)          # 570 - 598 m 
    gradRho[2] = np.mean(np.gradient(np.mean(Rho_lower[i: i+4, :], axis = 0),
                                     depths_lower))                                               # 597 - 715 m 
    # print(Rho_lower[i: i+4, :])
    # print()
    
    ################ grad Rho ####################

    meanRho = np.ones([3])

    meanRho[0] = (np.mean(Rho520[i: i+4], axis = 0) + np.mean(Rho571[i: i+4], axis = 0)) / (2)          # 526 - 566 m 
    meanRho[1] = (np.mean(Rho571[i: i+4], axis = 0) + np.mean(Rho597[i: i+4], axis = 0)) / (2)          # 570 - 598 m 
    meanRho[2] = np.mean(np.mean(Rho_lower[i: i+4, :], axis = 0))  
    
    ################# Ri #######################

    Ri_i = np.abs( 9.81 / (meanRho) * (gradRho) / (gradU)**2 )
    Ri_bulk[i-idxStart] = Ri_i
    # N_2_[i-idxStart] = N_2
    maxRi[i-idxStart] = np.max(Ri_i)
f, ax = plt.subplots(1, 1, figsize=(12, 3))


time_plot = Time2[idxStart: idxStart + TimeLen]
time4interpolation = np.linspace(0, 100, time_plot.shape[0])


print(time_plot.shape)
print(Ri_bulk[:, 0])
print(Ri_bulk.shape)

for i in range(3):
    Ri_i = Ri_bulk[:, i]
    y = Ri_bulk[~np.isnan(Ri_i), i]
    x = time4interpolation[~np.isnan(Ri_i)]
    Ri_bulk[:, i] = interp.interp1d(x, y)(time4interpolation)

colors = ['r', 'b', 'lime']

ax.plot(time_plot, Ri_bulk[:, 0], label = "Ri (526 - 566 m)", zorder = 10, c = colors[0])
ax.plot(time_plot, Ri_bulk[:, 1], label = "Ri (570 - 598 m)", c = colors[1])
ax.plot(time_plot, Ri_bulk[:, 2], label = "Ri (597 - 715 m)", c = colors[2])

print(np.mean(Ri_bulk[:, 0]))
print(np.mean(Ri_bulk[:, 1]))
print(np.mean(Ri_bulk[:, 2]))

ax.legend()
ax.set_ylabel("Ri")
ax.axhline(1/4, c = 'k')
ax.text(Time2[100], 0.01, 'Ri = 1/4', color = 'k')
ax.set_yscale('log')
plt.tight_layout()
ax.set_xlim([pd.to_datetime("18.02.2009T17:00"), 
             pd.to_datetime("20.02.2009T17:00")])
# plt.savefig("Ri_bulk_1000.png", dpi = 600)
plt.show()
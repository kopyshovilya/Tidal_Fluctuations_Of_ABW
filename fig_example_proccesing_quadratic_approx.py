import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import leastsq
import matplotlib.ticker as ticker


dir = "data/M3_UV.tab"
df = pd.read_table(dir, sep='\t', skiprows = 20)
LAT = -74.510500
phi = -70 / 360 * 2 * np.pi
SA4T = 6
LA4T = 1

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


fig = plt.figure(figsize=(12, 6), constrained_layout=True)
widths = [1]
heights = [1, 1, 1]
gs = fig.add_gridspec(ncols=1, nrows=3, width_ratios=widths, height_ratios=heights)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

idxStart = 10
TimeLen = 500

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

SA = 48 * 60 // 5
depSensors = np.array([715,648,623,597,571,520,468,417,365])



for i in tqdm(range(idxStart, idxStart + TimeLen)):

    U_along, U_cross = (U2_along[i], U2_cross[i])

    Cur = np.sqrt(U_along**2 + U_cross**2)
    Cur_init = Cur.copy()

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


LA = 1 * 60 // 5
coreLoc = np.convolve(coreLoc, np.ones(LA) / LA, mode='same')
ax3.plot(Time2[idxStart: idxStart + TimeLen], -coreLoc, 'gold')

data = np.zeros(stream_parabolic.shape)
data[stream_parabolic > 1] = 1
data[stream_parabolic > 2] = 2
data[stream_parabolic > 3] = 3
# plt.imshow(stream.T)
Speed_cross = np.zeros(stream_parabolic.shape)
Speed_cross[stream_cross_quadr > 0] = 1
Speed_cross[stream_cross_quadr < 0] = -1

Speed_along = np.zeros(stream_parabolic.shape)
Speed_along[stream_along_quadr > 0.5] = 1
Speed_along[stream_along_quadr < 0.5] = -1

g1 = ax1.pcolormesh(Time2[idxStart: idxStart + TimeLen], -depSensors2, stream.T,           vmin = 0.5, vmax = 1.5)
g2 = ax2.pcolormesh(Time2[idxStart: idxStart + TimeLen], -depSensors2, stream_parabolic.T, vmin = 0.5, vmax = 1.5)
g3 = ax3.pcolormesh(Time2[idxStart: idxStart + TimeLen], -depSensors2, data.T,             vmin = 0.5, vmax = 1.5)


Time2_grid, depGrid = np.meshgrid(Time2[idxStart: idxStart + TimeLen:6], -depSensors2[::3])



########################################################################

plt.colorbar(mappable=g2)


ax1.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))
ax2.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))
ax3.set_xlim(np.array(['2009-02-18T17:00', '2009-02-20T17:00'], dtype = 'datetime64'))

ax1.set_ylim([-605, -525])
ax2.set_ylim([-605, -525])
ax3.set_ylim([-605, -525])

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

ax3.xaxis.get_ticklocs(minor=True)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax3.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))

ax1.set_ylabel("Depth, m", fontsize = 14)
ax2.set_ylabel("Depth, m", fontsize = 14)
ax3.set_ylabel("Depth, m", fontsize = 14)

ax3.set_xlabel("Time", fontsize = 14)




f, ax = plt.subplots(1, 1, figsize = (2, 4))

idxStart = 10
TimeLen = 500
ax.set_ylim([605, 525])

for idx in range(415, 416):
    t = Time2[idxStart + idx]
    u_init = stream[idx, :]
    u_proc = stream_parabolic[idx, :]
    plt.plot(u_init, depSensors2, 'ko')
    plt.plot(u_proc, depSensors2, label = t, c='r')

plt.xlabel('U')
plt.ylabel('depth, m')

plt.tight_layout()
# plt.savefig('Example_of_processing_current_plot.png', dpi = 600, transparent = 1)
plt.show()


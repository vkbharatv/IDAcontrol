# %%
from IDA_ControlSystem import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["font.serif"] = ["Times New Roman", "Times New Roman"]

s = ctrl.TransferFunction.s
G = 1 / ((s + 1) * (0.7 * s + 1))
theta = 2
setpoint = 1.0
sim_time = 100.0
dt = 0.01
time_steps = int(sim_time / dt)
polset = -np.real(ctrl.poles(G))
process = ProcessDefinition(G, theta, dt=dt)
spf = ProcessDefinition((1 / (1 + s)), 0, dt=dt)
pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, dt=dt)
fig, ax = plt.figure(), plt.gca()
y_T = []
l = np.linspace(0.05, 0.09, 10)  # Different values of psi between 0 to 1
psi = []
for l_i in l:
    pid.IIMC_tuning(process, theta=theta, l=l_i)
    process.reset()
    pid.reset()
    spf.reset()
    for i in range(time_steps):
        current_output = process.y[-1][0]
        spf.step(setpoint)
        if i > int(0.5 * time_steps):
            D = 0.1  # Disturbance
        else:
            D = 0
        control_signal = pid.compute(spf.y[-1][0], current_output) + D
        process.step(control_signal)
    y_T.append(process.y)
    psi.append(pid.psi)

#  Plot results
plt.gca()
for i, psi_val in enumerate(psi):
    plt.plot(process.t, y_T[i], label=f"$\\psi={psi_val:.2f}$")

plt.plot(process.t, spf.y, "r--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Process Output with IIMC-PID Control")
plt.legend(ncol=2)
plt.grid()
plt.show()

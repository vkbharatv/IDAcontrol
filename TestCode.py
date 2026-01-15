# %%
from IDA_ControlSystem import *
import numpy as np
import matplotlib.pyplot as plt

s = ctrl.TransferFunction.s
G = 1 / (s + 1)
theta = 3.7
setpoint = 1.0
sim_time = 100.0
dt = 0.001
time_steps = int(sim_time / dt)
polset = -np.real(ctrl.poles(G))
print(polset)
process = ProcessDefinition(G, theta, dt=dt)
pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, dt=dt)
pid.IIMC_tuning(process, theta=theta, l=0.21)



for _ in range(time_steps):
    current_output = process.y[-1][0]
    control_signal = pid.compute(setpoint, current_output)
    process.step(control_signal)

# %% Plot results
plt.figure()
plt.plot(process.t, [y[0] for y in process.y], label="Process Output")
plt.plot(process.t, [setpoint] * len(process.t), "r--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Process Output with PID Control")
plt.legend()
plt.grid()
plt.savefig("process_output_pid_control.png", dpi=300)

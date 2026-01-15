# %%
"""
Delay Control System Simulation
This module defines a DelayControlSystem class that simulates a control system with input delay. It includes methods for stepping through the simulation, resetting the system, checking marginal stability, and calculating gain and phase margins.
Developed for educational and research purposes.
Author: Dr. Bharat Verma
Date: 2026-10-01
Institution: The LNMIIT, Jaipur, India
ORCID: https://orcid.org/0000-0001-7600-7872
"""

# %%
from unittest import case
from unittest import case
import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


# %%
class ProcessDefinition:

    def __init__(self, tf, delay_time, dt=0.01, Total_time=10):
        self.plant = tf
        self.delay_time = delay_time
        self.A, self.B, self.C, self.D = ctrl.ssdata(ctrl.tf2ss(self.plant))
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self.sim_time = Total_time
        self.dt = dt
        # Use dynamic storage so arrays grow as needed.
        self.t = [0.0]
        self.u = [0.0]  # Time series of actual inputs (with delay applied)
        self.y = [np.zeros(self.p)]
        self.x = [np.zeros(self.n)]
        self.i = 0
        # Delay buffer for storing past inputs
        self.delay_steps = int(np.ceil(self.delay_time / self.dt)) if delay_time > 0 else 0
        self.u_delay_buffer = np.zeros(self.delay_steps) if self.delay_steps > 0 else np.array([])
        self.buffer_index = 0

    def reset(self):
        self.delay_time = 0
        self.t = [0.0]
        self.u = [0.0]
        self.y = [np.zeros(self.p)]
        self.x = [np.zeros(self.n)]
        self.i = 0
        self.buffer_index = 0
        if self.delay_steps > 0:
            self.u_delay_buffer = np.zeros(self.delay_steps)

    def step(self,u=1.0):
        self.i = self.i + 1
        dt = self.dt 
        next_t = self.t[-1] + dt

        # Get current input from input signal
        u_current_input = np.asarray(u).item() if np.asarray(u).size == 1 else np.asarray(u).flatten()[0]

        # Apply delay using buffer
        if self.delay_steps > 0:
            self.u_delay_buffer[self.buffer_index] = u_current_input
            u_current = self.u_delay_buffer[(self.buffer_index + 1) % self.delay_steps]
            self.buffer_index = (self.buffer_index + 1) % self.delay_steps
        else:
            u_current = u_current_input

        x_prev = self.x[-1]
        x_next = x_prev + dt * (self.A @ x_prev + np.asarray(self.B).flatten() * u_current)

        # Output: y = C*x + D*u
        y_next = self.C @ x_next + np.asarray(self.D).flatten()[0] * u_current

        self.t.append(next_t)
        self.x.append(x_next)
        self.y.append(y_next)
        self.u.append(u_current)
        return x_next, y_next, next_t

    def marginal_stability(self):
        eigvals = np.linalg.eigvals(self.A)
        for val in eigvals:
            if np.real(val) > 0:
                return False
        return True

    def gain_phase_margin(self, omega=np.logspace(-3, 3, 10000)):
        num = self.plant.num[0][0]
        den = self.plant.den[0][0]
        tau = self.delay_time

        # Define the frequency range for analysis
        s = 1j * omega

        # Calculate the transfer function L(s) = N(s) / D(s) * e^(-tau * s)
        L = (np.polyval(num, s) / np.polyval(den, s)) * np.exp(-tau * s)

        # Calculate magnitude and phase
        magnitude = ctrl.mag2db(np.abs(L))
        phase = np.angle(L, deg=True)
        phase = np.unwrap(np.deg2rad(phase)) * (180 / np.pi)

        # Find gain crossover frequency (where magnitude is 0 dB)
        gain_crossover_idx = np.where(np.isclose(magnitude, 0, atol=0.01))[0]
        if len(gain_crossover_idx) == 0:
            phase_margin = np.inf
            gain_crossover_freq = None
        else:
            gain_crossover_freq = omega[gain_crossover_idx[0]]
            phase_margin = 180 + phase[gain_crossover_idx[0]]

        # Find phase crossover frequency (where phase is -180 degrees)
        phase_crossover_idx = np.where(np.isclose(phase, -180, atol=1))[0]
        if len(phase_crossover_idx) == 0:
            gain_margin = np.inf
            phase_crossover_freq = None
        else:
            phase_crossover_freq = omega[phase_crossover_idx[0]]
            gain_margin = 0 - magnitude[phase_crossover_idx[0]]

        return gain_margin, phase_margin, gain_crossover_freq, phase_crossover_freq


class PIDController:

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01, N=100):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.N = N  # Derivative filter coefficient
        self.derivative_filtered = 0.0

    def __call__(self):
        print("The PID controller is")
        print(f"Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}")

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        self.derivative_filtered = (
            (self.N * self.dt * derivative + self.derivative_filtered)
            / (self.N * self.dt + 1)
            if hasattr(self, "derivative_filtered")
            else derivative
        )
        output = (
            (self.Kp * error)
            + (self.Ki * self.integral)
            + (self.Kd * self.derivative_filtered)
        )
        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_filtered = 0.0

    def update_gains(self, Kp, Ki, Kd):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)

    def IIMC_tuning(self, process: ProcessDefinition,theta=0.01,l=0.5):
        # Get process parameters
        num = process.plant.num[0][0]
        den = process.plant.den[0][0]
        polset = -np.real(ctrl.poles(process.plant))


        print(len(den))
        if len(den) == 2:
            K = num[0] / den[1]
            tau = den[0] / den[1]
            b=tau
            a = 0
            order = 1
        if len(den) == 3:
            K = num[0] / den[2]
            a = den[0] / den[2]
            b = den[1] / den[2]
            order = 2
        else :
            print(f"Order: {order} K: {K} a: {a} b: {b}")
            # raise ValueError("IIMC tuning is only implemented for first and second order systems.")
        theta = theta
        psi = l * (min(polset)+0.5/theta)-(0.5/theta)
        print(f"Process Gain K: {K}, a: {a}, b: {b}, psi: {psi}")

        if order == 1:
            tau_hat = tau/(1-tau*psi)
            K_hat = K*np.exp(psi*theta)/(1 - tau*psi)
            kc = (tau_hat+0.5*theta)/(theta*K_hat)
            ti = tau_hat + 0.5*theta
            td = (0.5*tau_hat*theta)/(tau_hat+0.5*theta)

        elif order == 2:
            K_hat = K*np.exp(psi*theta)/(1 - b*psi+ a*psi**2)
            a_hat = a/(1 - b*psi + a*psi**2)
            b_hat = (b - 2*a*psi)/(1 - b*psi + a*psi**2)
            kc= b_hat/(K_hat*theta) 
            ti= b_hat
            td= (a_hat)/(b_hat)
        else:
            raise ValueError("IIMC tuning is only implemented for first and second order systems.")

        Kp = kc
        Ki = kc/ti
        Kd = kc*td
        self.update_gains(Kp, Ki, Kd)
        return Kp, Ki, Kd

if __name__ == "__main__":
    # Example usage
    plant_tf = ctrl.tf([10], [1, 2, 1])  # Example transfer function
    delay_time = 0.5  # 0.5 seconds delay
    process = ProcessDefinition(plant_tf, delay_time)

    pid = PIDController(Kp=2.0, Ki=1.0, Kd=1)
    pid.IIMC_tuning(process, theta=delay_time, psi=0.2)
    setpoint = 1.0
    sim_time = 10.0
    dt = 0.01
    time_steps = int(sim_time / dt)

    for _ in range(time_steps):
        current_output = process.y[-1][0]
        pid.update_gains(3.0, 1.0, 0.5)  # Update PID gains if needed
        control_signal = pid.compute(setpoint, current_output)
        process.step(control_signal)

    # Plot results
    plt.figure()
    plt.plot(process.t, [y[0] for y in process.y], label="Process Output")
    plt.plot(process.t, [setpoint] * len(process.t), "r--", label="Setpoint")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title("Process Output with PID Control")
    plt.legend()
    plt.grid()
    plt.show()

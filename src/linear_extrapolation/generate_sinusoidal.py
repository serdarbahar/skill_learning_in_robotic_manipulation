import numpy as np

def generate_sinusoidal(num_sinus=1, num_points=100, duration=1, amplitude=[1], frequency=[1], phase=[0]):

    assert len(amplitude) == num_sinus, "Length of amplitude list must match num_sinus"
    assert len(frequency) == num_sinus, "Length of frequency list must match num_sinus"
    assert len(phase) == num_sinus, "Length of phase list must match num_sinus"

    t = np.linspace(0, duration, num_points)
    data = np.zeros((num_sinus, num_points))
    for i in range(num_sinus):
        data[i, :] = amplitude[i] * np.sin(2 * np.pi * frequency[i] * t + phase[i])
    return data

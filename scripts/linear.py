from linear_extrapolation.linear_extrapolator import LinearExtrapolator
from linear_extrapolation.visualizer import Visualizer
from linear_extrapolation.generate_sinusoidal import generate_sinusoidal
import numpy as np

if __name__ == "__main__":

    lnr_extrp = LinearExtrapolator(num_points=100)

    # sinusoidal data
    num_traj = 2
    S = np.array([0.5, 1.0])

    data = generate_sinusoidal(num_sinus=num_traj, amplitude=S, frequency=[0.5, 1.0], phase=[0.0, 0.0])

    lnr_extrp.fit(data, S)

    test_s = np.array([-0.2, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4])  # normalized scalar values for extrapolation

    pred = lnr_extrp.extrapolate(test_s)

    visualizer = Visualizer(x=data, y=pred)
    visualizer.plot()
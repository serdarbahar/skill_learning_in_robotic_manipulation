from linear_extrapolation.linear_basic_extrapolator import LinearBasicExtrapolator
from linear_extrapolation.visualizer import Visualizer
from linear_extrapolation.generate_sinusoidal import generate_sinusoidal
import numpy as np

if __name__ == "__main__":

    lnr_extrp = LinearBasicExtrapolator(num_points=100)

    # sinusoidal data
    num_traj = 2

    data = generate_sinusoidal(num_sinus=num_traj, amplitude=[0.5, 1.0], frequency=[1.0, 1.0], phase=[0.0, 0.0])

    lnr_extrp.fit(data)

    test_s = np.array([1.5])

    pred = lnr_extrp.extrapolate(test_s)

    visualizer = Visualizer(x=data, y=pred)
    visualizer.plot()
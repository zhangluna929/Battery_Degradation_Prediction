import numpy as np

from battery_degradation.features.electrochem import incremental_capacity, differential_voltage


def test_ic_and_dva_shapes():
    voltage = np.linspace(4.2, 3.0, 100)
    capacity = np.linspace(0, 1.5, 100)
    ic = incremental_capacity(voltage, capacity, smooth=False)
    dva = differential_voltage(voltage, capacity, smooth=False)
    assert ic.shape == voltage.shape
    assert dva.shape == voltage.shape 
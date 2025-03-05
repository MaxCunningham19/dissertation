import numpy as np

from .Boltzmann import Boltzmann


class DecayBoltzmann(Boltzmann):
    def __init__(self, temperature=10.0, temperature_decay=0.99, temperature_min=0.5):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    def update_parameters(self):
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)

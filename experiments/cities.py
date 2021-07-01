from dataclasses import dataclass
import numpy as np


@dataclass
class Ciudad:
    nombre: str
    lat_lo: np.float64
    lat_hi: np.float64
    lng_lo: np.float64
    lng_hi: np.float64


ciudades = [
    Ciudad(
        nombre='Guadalajara',
        lat_lo=20.573875,
        lat_hi=20.767892,
        lng_lo=-103.496963,
        lng_hi=-103.199645,
    )
]

from scipy import constants
import numpy as np

def calcPlasmaFreq(ne: np.ndarray,
                   e=constants.elementary_charge,
                   epsilon=constants.epsilon_0,
                   me=constants.electron_mass) -> np.ndarray:
    """Calculates the plasma frequency.

    Args:
        ne (numpy.ndarray): Electron number density array
        e (float): Elementary charge (default: real value in SI)
        epsilon (float): Vacuum permitivity (default: real value in SI)
        me (float): Electron mass (default: real value in SI)

    Returns:
        omega (numpy.ndarray): Plasma frequency
    """
    return np.sqrt(ne*e**2/(epsilon*me))

def calcPressureParPer(Pij: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculates parallel and perpendicular pressure.
    
    Args:
        Pij (numpy.ndarray): Pressure tensor
        B (numpy.ndarray): Magnetic field vector

    Returns:
        Ppp (numpy.ndarray): Parallel (component 0) and perpendicular
            (component 1) pressure
    """
    Pxx = Pij[..., 0, np.newaxis]
    Pxy = Pij[..., 1, np.newaxis]
    Pxz = Pij[..., 2, np.newaxis]
    Pyy = Pij[..., 3, np.newaxis]
    Pyz = Pij[..., 4, np.newaxis]
    Pzz = Pij[..., 5, np.newaxis]

    b = np.sqrt(B[..., 0, np.newaxis]*B[..., 0, np.newaxis] 
                + B[..., 1, np.newaxis]*B[..., 1, np.newaxis]
                + B[..., 2, np.newaxis]*B[..., 2, np.newaxis])
    bx = B[..., 0, np.newaxis] / b
    by = B[..., 1, np.newaxis] / b
    bz = B[..., 2, np.newaxis] / b

    Ppp = np.copy(Pij[..., 0:2])
    Ppp[..., 0] = bx*bx*Pxx + by*by*Pyy + bz*bz*Pzz \
                  + 2.0*(bx*by*Pxy + bx*bz*Pxz + by*bz*Pyz) # parallel
    Ppp[..., 1] = (Pxx + Pyy + Pzz - Ppar) / 2.0 # perpendicular
    return Ppp

def calcAgyrotropy(Pij: np.ndarray, B: np.ndarray,
                   measure=None) -> np.ndarray:
    """Calculates agyrotropy

    The result is a 6-component field, unless the measure is
    selected. Then it calculates a 1-component vector based Frobenius
    or Swisdak.
    
    Args:
        Pij (numpy.ndarray): Pressure tensor
        B (numpy.ndarray): Magnetic field vector
        measure (str): Calculate an agyrotropy measure instead 
            ('Frobenius' or 'Swisdak')

    Returns:
        agyro (numpy.ndarray): Agyrotropy

    """
    Pxx = Pij[..., 0, np.newaxis]
    Pxy = Pij[..., 1, np.newaxis]
    Pxz = Pij[..., 2, np.newaxis]
    Pyy = Pij[..., 3, np.newaxis]
    Pyz = Pij[..., 4, np.newaxis]
    Pzz = Pij[..., 5, np.newaxis]

    b = np.sqrt(B[..., 0, np.newaxis]**2
                + B[..., 1, np.newaxis]**2
                + B[..., 2, np.newaxis]**2)
    bx = B[..., 0, np.newaxis] / b
    by = B[..., 1, np.newaxis] / b
    bz = B[..., 2, np.newaxis] / b

    Ppar = bx*bx*Pxx + by*by*Pyy + bz*bz*Pzz  \
           + 2.0*(bx*by*Pxy + bx*bz*Pxz + by*bz*Pyz)
    Pper = (Pxx + Pyy + Pzz - Ppar) / 2.0

    I1 = Pxx + Pyy + Pzz
    I2 = Pxx*Pyy + Pxx*Pzz + Pyy*Pzz - (Pxy*Pxy + Pxz*Pxz + Pyz*Pyz)

    Pixx = Pxx - (Ppar*bx*bx + Pper*(1 - bx*bx))
    Pixy = Pxy - (Ppar*bx*by + Pper*(0 - bx*by))
    Pixz = Pxz - (Ppar*bx*bz + Pper*(0 - bx*bz))
    Piyy = Pyy - (Ppar*by*by + Pper*(1 - by*by))
    Piyz = Pyz - (Ppar*by*bz + Pper*(0 - by*bz))
    Pizz = Pzz - (Ppar*bz*bz + Pper*(1 - bz*bz))

    if measure is None:
        agyro = np.copy(Pij[..., 0:6])
        agyro[..., 0] = Pixx
        agyro[..., 1] = Pixy
        agyro[..., 2] = Pixz
        agyro[..., 3] = Piyy
        agyro[..., 4] = Piyz
        agyro[..., 5] = Pizz
        return agyro
    elif measure.lower() == "swisdak":
        return np.sqrt(1 - 4*I2 / ((I1 - Ppar)*(I1 + 3*Ppar)))
    elif measure.lower() == "frobenius":
        return np.sqrt(Pixx**2 + 2*Pixy**2 + 2*Pixz**2 
                       + Piyy**2 + 2*Piyz**2 + Pizz**2) \
            / np.sqrt(2*Pper**2 + 4*Ppar*Pper)

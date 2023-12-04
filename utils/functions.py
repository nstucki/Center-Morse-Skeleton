import numpy as np
from scipy.ndimage import distance_transform_edt

def distance_transform(image):
    distance_f = np.copy(image)
    distance_f = distance_transform_edt(distance_f)
    distance_b = np.copy(1-image)
    distance_b = distance_transform_edt(distance_b)
    signed_distance = -distance_f + distance_b

    return signed_distance

def minimum_distance_in(image):
    min_distances = []
    values = np.sort(np.unique(image))
    min_distance = np.min(np.abs(values[1:]-values[:-1]))

    return min_distance
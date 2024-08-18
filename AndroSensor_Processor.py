import csv
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, asin, atan2, degrees, radians
from numpy import linalg


def rotate_3D_tait_bryan(
        coordinates: list[tuple[float, float, float]],
        alpha_radians: float,
        beta_radians: float,
        gamma_radians: float
    ) -> list[tuple[float, float, float]]:

    transformed_coordinates: list[tuple[float, float, float]] = []

    rotation_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (
            cos(alpha_radians)*cos(beta_radians),
            cos(alpha_radians)*sin(beta_radians)*sin(gamma_radians) - sin(alpha_radians)*cos(gamma_radians),
            cos(alpha_radians)*sin(beta_radians)*cos(gamma_radians) + sin(alpha_radians)*sin(gamma_radians)
        ),
        (
            sin(alpha_radians)*cos(beta_radians),
            sin(alpha_radians)*sin(beta_radians)*sin(gamma_radians) + cos(alpha_radians)*cos(gamma_radians),
            sin(alpha_radians)*sin(beta_radians)*cos(gamma_radians) - cos(alpha_radians)*sin(gamma_radians)
        ),
        (
            -sin(beta_radians),
            cos(beta_radians)*sin(gamma_radians),
            cos(beta_radians)*cos(gamma_radians)
        )
    )

    transformed_coordinate: tuple[float, float, float]
    for coordinate in coordinates:
        transformed_coordinate = tuple(linalg.solve(
            rotation_matrix,
            (coordinate[0], coordinate[1], coordinate[2])
        ))
        transformed_coordinates.append(transformed_coordinate)

    return transformed_coordinates


def smooth_points(
        coordinates: list[tuple[float, float, float]],
        points_to_average: int
    ) -> list[tuple[float, float, float]]:

    smoothed_coordinates: list[tuple[float, float, float]] = []
    smoothed_coordinate: tuple[float, float, float]

    if points_to_average > 1:
        for coordinate_index in range(len(coordinates)):
            average_start_index: int = max(0, coordinate_index - int(points_to_average/2))
            average_end_index: int = min(len(coordinates)-1, coordinate_index + int(points_to_average/2 + 1))
            smoothed_coordinate = tuple(
                sum(
                    coordinate[axis]
                    for coordinate
                    in coordinates[average_start_index:average_end_index]
                ) / points_to_average
                for axis
                in range(3)
            ) # type: ignore
            smoothed_coordinates.append(smoothed_coordinate)

    else:
        smoothed_coordinates = coordinates

    return smoothed_coordinates


def find_index(string: str, list_to_look_in: list[str]) -> int:
    """
    finds the index in the list which contains string
    """
    index = 0
    for item in list_to_look_in:
        if item.find(string) != -1:
            break
        index += 1

    return index


def calculate_medians(coordinates: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    number_of_coordinates: int = len(coordinates)
    sorted_x_coordinates: list[float] = sorted(
        coordinate[0]
        for coordinate
        in coordinates
    )
    sorted_y_coordinates: list[float] = sorted(
        coordinate[1]
        for coordinate
        in coordinates
    )
    sorted_z_coordinates: list[float] = sorted(
        coordinate[2]
        for coordinate
        in coordinates
    )
    means: tuple[float, float, float] = (
        sorted_x_coordinates[int(number_of_coordinates/2 + 1)],
        sorted_y_coordinates[int(number_of_coordinates/2 + 1)],
        sorted_z_coordinates[int(number_of_coordinates/2 + 1)]
    )

    return means


print("defining variables...")

filename: str = input("Paste AndroSensor file path here:\n")

average_n_values = 1 # number of points to average into a single point on the graphs

print("loading file into memory...")

with open(filename, 'r', encoding = "UTF-8") as file:
    reader = csv.reader(file)
    rows = [row for row in reader] # read the whole file

print("finding indices...")

# find column indices of variables
acceleration_indices: tuple[int, int, int] = (
    find_index("ACCELEROMETER X (m/s²)", rows[0]),
    find_index("ACCELEROMETER Y (m/s²)", rows[0]),
    find_index("ACCELEROMETER Z (m/s²)", rows[0])
)

milliseconds_index = find_index("Time since start in ms ", rows[0])

print("copying data into lists...")

phone_accelerations: list[tuple[float, float, float]] = []
phone_acceleration_magnitudes: list[float] = []
seconds_since_starts: list[float] = []

for row in rows[1:]:
    seconds_since_start: float = float(row[milliseconds_index])/1000

    phone_acceleration: tuple[float, float, float] = tuple(
        float(row[acceleration_index])
        for acceleration_index
        in acceleration_indices
    ) # type: ignore

    # append data to lists
    seconds_since_starts.append(seconds_since_start)

    phone_accelerations.append(phone_acceleration)

    phone_acceleration_magnitudes.append(sqrt(sum(acceleration**2 for acceleration in phone_acceleration)))

print("{:.1f} datapoints per second".format(len(rows)/seconds_since_starts[-1]))

print("calculating matrix transformation...")

median_phone_acceleration: tuple[float, float, float] = calculate_medians(phone_accelerations)
print("median phone acceleration: {}".format(median_phone_acceleration))
# median phone acceleration should essentially just be acceleration due to gravity
gravity_magnitude: float = sqrt(sum([i**2 for i in median_phone_acceleration]))
print("median phone acceleration magnitude: {}".format(gravity_magnitude))
unit_average_gravity = tuple([i/gravity_magnitude for i in median_phone_acceleration]) # direction of +Z
# this is the direction in which world +Z is in terms of phone axes

# now rotate accelerations such that gravity points in -Z direction
# https://en.wikipedia.org/wiki/Rotation_matrix
# using Tait-Bryan angles
# TL;DR: alpha, beta, gamma are yaw, pitch and roll respectively
# in other words, alpha is rotation about Z-axis, beta is rotation about Y-axis, and gamma is rotation about X-axis
alpha: float = 0.0 # rotation about Z-axis
beta: float = asin(unit_average_gravity[0]) # rotation about Y-axis
gamma: float = -atan2(unit_average_gravity[1], unit_average_gravity[2]) # rotation about X-axis

print("rotating about X-axis by {:.2f} degrees".format(degrees(gamma)))
print("rotating about Y-axis by {:.2f} degrees".format(degrees(beta)))
print("rotating about Z-axis by {:.2f} degrees".format(degrees(alpha)))

car_accelerations: list[tuple[float, float, float]] = rotate_3D_tait_bryan(phone_accelerations, alpha, beta, gamma)

median_car_acceleration: tuple[float, float, float] = calculate_medians(car_accelerations)

print("median acceleration in car coordinates: {}".format(median_car_acceleration))

# rotate around Z axis to line up rotated phone X and Y axes with car X and Y axes
alpha = 0.0
beta = 0.0 # rotation about Y-axis
gamma = 0.0 # rotation about X-axis

print("rotating about X-axis by {:.2f} degrees".format(degrees(gamma)))
print("rotating about Y-axis by {:.2f} degrees".format(degrees(beta)))
print("rotating about Z-axis by {:.2f} degrees".format(degrees(alpha)))

fixed_car_accelerations: list[tuple[float, float, float]] = rotate_3D_tait_bryan(car_accelerations, alpha, beta, gamma)

median_fixed_car_accelerations: tuple[float, float, float] = calculate_medians(fixed_car_accelerations)

print("median acceleration in car coordinates after manual rotation about Z: {}".format(median_fixed_car_accelerations))

print("averaging values over {} points ({:.2f} seconds)".format(
    average_n_values,
    average_n_values/(len(rows)/seconds_since_starts[-1])
))

smoothed_fixed_car_accelerations: list[tuple[float, float, float]] = smooth_points(fixed_car_accelerations, average_n_values)

print("plotting...")

# Plot 0: X, Y and Z acceleration on the same plot
plt.figure(0)
for axis in range(3):
    plt.scatter(
        seconds_since_starts,
        [smoothed_fixed_car_acceleration[axis] for smoothed_fixed_car_acceleration in smoothed_fixed_car_accelerations],
        s=3
    )

plt.xlabel("Time from Start (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Toyota Corolla Automatic 2009 Coastdown Test")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

plt.legend(("X", "Y", "Z"))

print("done")

plt.show()

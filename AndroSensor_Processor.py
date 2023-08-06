import csv
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, acos
from numpy import linalg, dot

print("defining variables...")

filename = "Sensor_record_20230805_092308_AndroSensor.csv"

average_n_values = 20 # number of points to average into a single point on the graphs

X_accelerations: list[float] = []
Y_accelerations: list[float] = []
Z_accelerations: list[float] = []

X_car_accelerations: list[float] = []
Y_car_accelerations: list[float] = []
Z_car_accelerations: list[float] = []

sum_accelerations: list[float] = []
seconds_since_starts: list[float] = []


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

print("loading file into memory...")

with open(filename, 'r', encoding = "UTF-8") as file:
    reader = csv.reader(file)
    rows = [row for row in reader] # read the whole file

print("finding indices...")

# find column indices of variables
X_acceleration_index = find_index("ACCELEROMETER X (m/s²)", rows[0])
Y_acceleration_index = find_index("ACCELEROMETER Y (m/s²)", rows[0])
Z_acceleration_index = find_index("ACCELEROMETER Z (m/s²)", rows[0])

milliseconds_index = find_index("Time since start in ms ", rows[0])

print("copying data into lists...")

for row in rows[1:]:
    seconds_since_start = float(row[milliseconds_index])/1000

    X_acceleration = float(row[X_acceleration_index])
    Y_acceleration = float(row[Y_acceleration_index])
    Z_acceleration = float(row[Z_acceleration_index])

    # append data to lists
    seconds_since_starts.append(seconds_since_start)

    X_accelerations.append(X_acceleration)
    Y_accelerations.append(Y_acceleration)
    Z_accelerations.append(Z_acceleration)

    sum_accelerations.append(sqrt(X_acceleration**2 + Y_acceleration**2 + Z_acceleration**2))

print("{:.1f} datapoints per second".format(len(rows)/seconds_since_starts[-1]))

print("calculating matrix transformation...")

average_gravity = (sum(X_accelerations)/len(X_accelerations), sum(Y_accelerations)/len(Y_accelerations), sum(Z_accelerations)/len(Z_accelerations))
gravity_magnitude = sqrt(sum([i**2 for i in average_gravity]))
unit_average_gravity = tuple([i/gravity_magnitude for i in average_gravity]) # direction of negative Z
# this is where world -Z is in terms of phone XYZ

# now rotate XYZ accelerations to put -Z in the direction of gravity
# https://en.wikipedia.org/wiki/Rotation_matrix
# using Talt-Bryan angles
gamma = 0 # rotation about Z-axis
beta = 0 # rotation about Y-axis
alpha = acos(dot(unit_average_gravity, (0, 0, 1))) # rotation about X-axis
print("rotating about X-axis by {:.2f} degrees".format(alpha*180/3.14159))
convert_phone_XYZ_into_car_XYZ_matrix = ((cos(beta)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma), cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)),
                                         (cos(beta)*sin(gamma), sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma), cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)),
                                         (-sin(beta), sin(alpha)*cos(beta), cos(alpha)*cos(beta)))

print("transforming all values...")

for i in range(len(X_accelerations)):
    X_car_acceleration, Y_car_acceleration, Z_car_acceleration = tuple(linalg.solve(convert_phone_XYZ_into_car_XYZ_matrix, (X_accelerations[i], Y_accelerations[i], Z_accelerations[i])))
    X_car_accelerations.append(X_car_acceleration)
    Y_car_accelerations.append(Y_car_acceleration)
    Z_car_accelerations.append(Z_car_acceleration)

if average_n_values > 1:
    print("averaging values over {} points ({:.2f} seconds)".format(average_n_values, average_n_values/(len(rows)/seconds_since_starts[-1])))

    for i in range(len(X_accelerations) - average_n_values):
        X_car_acceleration = sum(X_car_accelerations[i:i+average_n_values])/average_n_values
        Y_car_acceleration = sum(Y_car_accelerations[i:i+average_n_values])/average_n_values
        Z_car_acceleration = sum(Z_car_accelerations[i:i+average_n_values])/average_n_values    
        X_car_accelerations[i] = X_car_acceleration
        Y_car_accelerations[i] = Y_car_acceleration
        Z_car_accelerations[i] = Z_car_acceleration

print("plotting...")

# Plot 0: sum of XYZ acceleration
plt.figure(0)
plt.scatter(seconds_since_starts, sum_accelerations, s = 3)

plt.xlabel("Time from Start (s)")
plt.ylabel("Net Acceleration (m/s²)")
plt.title("Net Acceleration of Kia Sorento")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Plot 1: X, Y and Z acceleration on the same plot
plt.figure(1)
plt.scatter(seconds_since_starts, X_car_accelerations, s=3)
plt.scatter(seconds_since_starts, Y_car_accelerations, s=3)
plt.scatter(seconds_since_starts, Z_car_accelerations, s=3)

plt.xlabel("Time from Start (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration of Kia Sorento")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

plt.legend(("X", "Y", "Z"))

print("done")

plt.show()

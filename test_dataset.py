import pandas as pd
import numpy as np

data = pd.read_csv("active_satellites.csv")

# Earth's gravitational parameter
mu = 398600.4418

# Convert mean motion to radians per second
n = data['MEAN_MOTION'] * 2 * np.pi / 86400

# Calculate semi-major axis
a = (mu / (n**2))**(1/3)

# Calculate altitude
earth_radius = 6371
data['ALTITUDE_KM'] = a - earth_radius

print("\nSample altitude values:")
print(data[['OBJECT_NAME','ALTITUDE_KM']].head(10))


# classify orbit types
def classify_orbit(alt):
    if alt < 2000:
        return "LEO"
    elif alt < 35786:
        return "MEO"
    else:
        return "GEO"

data["ORBIT_TYPE"] = data["ALTITUDE_KM"].apply(classify_orbit)

print("\nOrbit type counts:")
print(data["ORBIT_TYPE"].value_counts())


import matplotlib.pyplot as plt

# ---- Graph 1: LEO zoom ----
plt.figure(figsize=(10,6))

leo_data = data[data["ALTITUDE_KM"] < 2000]

plt.hist(leo_data["ALTITUDE_KM"], bins=50)

plt.title("LEO Satellite Altitude Distribution (0–2000 km)")
plt.xlabel("Altitude (km)")
plt.ylabel("Number of Satellites")

plt.show()


# ---- Graph 2: Orbit type distribution ----
plt.figure(figsize=(8,5))

orbit_counts = data["ORBIT_TYPE"].value_counts()

plt.bar(orbit_counts.index, orbit_counts.values)

plt.title("Satellite Distribution by Orbit Type")
plt.xlabel("Orbit Type")
plt.ylabel("Number of Satellites")

plt.show()
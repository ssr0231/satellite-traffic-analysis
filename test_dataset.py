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



#------------------------------

from sklearn.cluster import KMeans

# Use altitude as feature
X = data[['ALTITUDE_KM']]

# Create model
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit model
data['CLUSTER'] = kmeans.fit_predict(X)

print("\nCluster counts:")
print(data['CLUSTER'].value_counts())


# ---- Professional Cluster Visualization ----

plt.figure(figsize=(10,6))

# sort by altitude for clean visualization
sorted_data = data.sort_values(by="ALTITUDE_KM")

# plot clusters
plt.scatter(
    range(len(sorted_data)),
    sorted_data["ALTITUDE_KM"],
    c=sorted_data["CLUSTER"],
    cmap='viridis',
    s=5
)

# horizontal lines for orbit boundaries
plt.axhline(2000, color='red', linestyle='--', label='LEO/MEO Boundary')
plt.axhline(35786, color='orange', linestyle='--', label='MEO/GEO Boundary')

plt.title("Satellite Clustering and Orbital Regions", fontsize=14)
plt.xlabel("Satellites (sorted by altitude)")
plt.ylabel("Altitude (km)")

plt.legend()

plt.ylim(0, 40000)  # focus on meaningful range

plt.grid(alpha=0.3)

plt.show()


# ---- Label clusters based on average altitude ----

cluster_means = data.groupby("CLUSTER")["ALTITUDE_KM"].mean()

cluster_labels = {}

for cluster, alt in cluster_means.items():
    if alt < 2000:
        cluster_labels[cluster] = "LEO"
    elif alt < 35786:
        cluster_labels[cluster] = "MEO"
    elif alt < 50000:
        cluster_labels[cluster] = "GEO"
    else:
        cluster_labels[cluster] = "OUTLIER"

data["CLUSTER_LABEL"] = data["CLUSTER"].map(cluster_labels)

print("\nCluster Label Mapping:")
print(cluster_labels)

#-----------bar graph ---------------
plt.figure(figsize=(8,5))

counts = data["CLUSTER_LABEL"].value_counts()

plt.bar(counts.index, counts.values)

plt.title("Satellite Distribution by Orbital Region", fontsize=13)
plt.xlabel("Orbit Type")
plt.ylabel("Number of Satellites")

plt.grid(axis='y', alpha=0.3)

plt.show()


# ---- Generate Simplified Satellite Positions ----

# simulate angular position
data["ANGLE"] = np.radians(data["MEAN_ANOMALY"])

# convert altitude to orbital radius
data["RADIUS"] = data["ALTITUDE_KM"] + 6371

# calculate x, y positions (2D approximation)
data["X"] = data["RADIUS"] * np.cos(data["ANGLE"])
data["Y"] = data["RADIUS"] * np.sin(data["ANGLE"])



# ---- Collision Risk Calculation ----

collision_data = []

threshold = 50  # km

sample_data = data.sample(500).reset_index(drop=True)

for i in range(len(sample_data)):
    for j in range(i+1, len(sample_data)):
        
        dx = sample_data.iloc[i]["X"] - sample_data.iloc[j]["X"]
        dy = sample_data.iloc[i]["Y"] - sample_data.iloc[j]["Y"]
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < threshold:
            
            # risk score (inverse distance)
            if distance == 0:
                continue  # skip unrealistic identical positions

            risk = 1 / (distance + 1)
            
            collision_data.append((i, j, distance, risk))

print("\nTotal risky pairs:", len(collision_data))


# sort by highest risk
collision_data = sorted(collision_data, key=lambda x: x[3], reverse=True)

print("\nTop 5 highest risk collisions:")
for pair in collision_data[:5]:
    print(pair)

    

# ---- Professional Collision Visualization (Styled) ----

plt.figure(figsize=(10,6))

# filter LEO satellites
leo_sample = sample_data[sample_data["ALTITUDE_KM"] < 2000]

# plot satellites (soft blue)
plt.scatter(
    leo_sample["X"], leo_sample["Y"],
    color='#4A90E2',  # soft blue
    s=20,
    alpha=0.6,
    label="LEO Satellites"
)

# plot top risky pairs
for i, j, d, r in collision_data[:30]:
    
    x1 = sample_data.iloc[i]["X"]
    y1 = sample_data.iloc[i]["Y"]
    
    x2 = sample_data.iloc[j]["X"]
    y2 = sample_data.iloc[j]["Y"]
    
    plt.plot(
        [x1, x2], [y1, y2],
        color='#E74C3C',  # red
        linewidth=1.5,
        alpha=0.7
    )

# highlight highest risk pairs (top 5)
for i, j, d, r in collision_data[:5]:
    
    x1 = sample_data.iloc[i]["X"]
    y1 = sample_data.iloc[i]["Y"]
    
    x2 = sample_data.iloc[j]["X"]
    y2 = sample_data.iloc[j]["Y"]
    
    plt.plot(
        [x1, x2], [y1, y2],
        color="#F90505",  # red highlight
        linewidth=2.5,
        alpha=0.9
    )

plt.title("Collision Risk Zones in LEO", fontsize=14, fontweight='bold')
plt.xlabel("X Position (km)")
plt.ylabel("Y Position (km)")

plt.legend()
plt.grid(alpha=0.3)

# zoom into LEO
plt.xlim(-10000, 10000)
plt.ylim(-10000, 10000)

plt.show()


# ---- Heatmap Grid Setup ----

grid_size = 500  # km
heatmap = {}

# ---- Fill Heatmap ----

for i, j, distance, risk in collision_data:
    
    x1 = sample_data.iloc[i]["X"]
    y1 = sample_data.iloc[i]["Y"]
    
    x2 = sample_data.iloc[j]["X"]
    y2 = sample_data.iloc[j]["Y"]
    
    # midpoint of collision
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # convert to grid index
    grid_x = int(mid_x // grid_size)
    grid_y = int(mid_y // grid_size)
    
    # store risk
    if (grid_x, grid_y) not in heatmap:
        heatmap[(grid_x, grid_y)] = 0
    
    heatmap[(grid_x, grid_y)] += risk

print("Heatmap cells:", len(heatmap))

# ---- Convert Heatmap to Matrix ----

# get all grid coordinates
x_coords = [key[0] for key in heatmap.keys()]
y_coords = [key[1] for key in heatmap.keys()]

# find boundaries
# use full satellite space instead of only collision space
min_x = int(sample_data["X"].min() // grid_size)
max_x = int(sample_data["X"].max() // grid_size)

min_y = int(sample_data["Y"].min() // grid_size)
max_y = int(sample_data["Y"].max() // grid_size)

# create empty matrix
heatmap_matrix = np.zeros((max_y - min_y + 1, max_x - min_x + 1))

# fill matrix
for (gx, gy), value in heatmap.items():
    heatmap_matrix[gy - min_y][gx - min_x] = value


print("Matrix shape:", heatmap_matrix.shape)


# ---- Smooth Heatmap (Gaussian Filter) ----

from scipy.ndimage import gaussian_filter

# apply smoothing
smooth_heatmap = gaussian_filter(heatmap_matrix, sigma=2)

# ---- Plot Smooth Heatmap ----

plt.figure(figsize=(8,6))

# define one common extent
extent = [
    min_x * grid_size, max_x * grid_size,
    min_y * grid_size, max_y * grid_size
]

plt.imshow(
    smooth_heatmap,
    origin='lower',
    cmap='hot',
    extent=extent
)

plt.contour(
    smooth_heatmap,
    extent=extent,
    colors='white',
    linewidths=0.5
)


plt.colorbar(label='Collision Risk Density')

plt.title("Spatial Collision Risk Density in LEO (Smoothed Heatmap)", fontsize=14)
plt.xlabel("Orbital Position X (km)")
plt.ylabel("Orbital Position Y (km)")

plt.scatter(
    sample_data["X"],
    sample_data["Y"],
    s=5,
    color='cyan',
    alpha=0.4,
    label="Satellites"
)

plt.legend()

plt.xlim(-8000, 8000)
plt.ylim(-8000, 8000)

plt.show()
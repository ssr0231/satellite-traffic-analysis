# Orbital Traffic Analysis and Collision Risk Optimization in Satellite Mega-Constellations

## Project Overview

This project focuses on analyzing satellite orbital data to detect potential collision risks and identify congestion patterns in Earth’s orbital regions, particularly Low Earth Orbit (LEO). It utilizes real-world satellite data and computational techniques to model orbital behavior, assess risk, and visualize high-density collision zones.

---

## Objectives

* Analyze large-scale satellite datasets to understand orbital distribution
* Identify congestion zones in different orbital regions
* Detect potential collision pairs based on spatial proximity
* Quantify collision risk using distance-based scoring
* Visualize spatial risk distribution using heatmaps

---

## Key Features

* Processing of real satellite data from CelesTrak
* Conversion of mean motion into orbital altitude using orbital mechanics
* Classification of satellites into LEO, MEO, and GEO
* Clustering of satellites using K-Means to identify orbital patterns
* Simulation of satellite positions in a 2D orbital plane
* Detection of potential collision events based on distance thresholds
* Calculation of collision risk scores
* Generation of smoothed spatial heatmaps for risk visualization

---

## Methodology

1. Data Acquisition
   Satellite data is collected from publicly available CelesTrak sources.

2. Orbital Computation
   Mean motion values are converted into orbital altitude using standard orbital mechanics formulas.

3. Orbit Classification
   Satellites are categorized into LEO, MEO, and GEO based on altitude ranges.

4. Clustering
   K-Means clustering is applied to identify patterns and group satellites by orbital characteristics.

5. Position Simulation
   Satellite positions are approximated in a 2D plane using angular and radial transformations.

6. Collision Detection
   Pairwise distances between satellites are computed, and potential collisions are identified using a defined threshold.

7. Risk Modeling
   Collision risk is quantified using an inverse distance-based scoring mechanism.

8. Visualization
   A Gaussian-smoothed heatmap is generated to represent spatial collision risk density across orbital regions.

---

## Technology Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SciPy

---

## Project Structure

satellite_collision/

* active_satellites.csv
* test_dataset.py
* README.md

---

## Results

The system successfully identifies high-density collision zones in LEO, demonstrating that collision risk is not uniformly distributed but concentrated in specific orbital regions. The heatmap visualization provides a clear spatial representation of these high-risk areas.

---

## Future Work

* Development of optimization algorithms for collision avoidance
* Extension to 3D orbital modeling
* Integration of real-time satellite tracking data
* Deployment of an interactive dashboard for visualization

---

## Author

Shubham Singh & Team

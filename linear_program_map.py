import pulp
import numpy as np
import networkx as nx
import osmnx as ox

class DSM:
    def __init__(self,graph,grid,potential_base):

# Environment configuration
        self.graph = graph
        self.nodes, self.edges = ox.graph_to_gdfs(graph)

        self.grid = grid

        self.potential_base = potential_base
        self.grid['centroid'] = self.grid.geometry.centroid.values
        self.grid['nearest_node_id'] = self.grid['centroid'].apply(
            lambda centroid: ox.distance.nearest_nodes(graph, centroid.x, centroid.y)
            )

    def Eu_distance_nodes(self,node1_id, node2_id):
        node1_data = self.graph.nodes[node1_id]
        node2_data = self.graph.nodes[node2_id]
        dx = node2_data['x'] - node1_data['x']
        dy = node2_data['y'] - node1_data['y']
        return (dx**2 + dy**2)**0.5
    
    def find_covered_base (self,incident_id,r):
        covered_base = []
        incident_node_id = self.grid.loc[incident_id,'nearest_node_id']
        for index, ambulance_base_node_id in zip(self.potential_base['index'], self.potential_base['nearest_node_id']):
            
            distance = self.Eu_distance_nodes(incident_node_id,ambulance_base_node_id)
            if distance <= r:
                covered_base.append(index)
        return covered_base

    def _get_neighborhood_set(self,r):
        neighborhood_set_dict = {}
        for incident_id in self.grid['index']:
            neighborhood_set = self.find_covered_base(incident_id,r)
            neighborhood_set_dict[incident_id] = neighborhood_set
        return neighborhood_set_dict


    def solve(self,r1=14664,r2=14664*1.5,alpha = 0.95,total_ambulances=None,max_abmulances = {}):

        model = pulp.LpProblem("Double_Standard_Model", pulp.LpMaximize)
        
        d = self.grid['mean_rate'].to_list()
        incident_id = self.grid['index'].to_list()
        base_id = self.potential_base['index'].to_list()
        # Decision variables
        y = pulp.LpVariable.dicts("y", base_id, lowBound=0, cat="Integer")     # number of ambulances at site j
        x1 = pulp.LpVariable.dicts("x1", incident_id, lowBound=0, upBound=1, cat="Binary")  # covered once
        x2 = pulp.LpVariable.dicts("x2", incident_id, lowBound=0, upBound=1, cat="Binary")  # covered twice

        # neighborhood set
        Wi1 = self._get_neighborhood_set(r1)
        Wi2 = self._get_neighborhood_set(r2)

        # Objective: maximize demand covered twice within r1
        model += pulp.lpSum(d[i] * x2[i] for i in incident_id)

        # constraints
        for i in incident_id:   # 30
            neighborhood_set = Wi2[i]
            model += pulp.lpSum(y[j] for j in neighborhood_set) >= 1

        model += pulp.lpSum(d[i] * x1[i] for i in incident_id) >= pulp.lpSum(alpha * d[i] for i in incident_id) #31

        for i in incident_id: #32
            neighborhood_set = Wi1[i]
            model += pulp.lpSum(y[j] for j in neighborhood_set) >= x1[i] + x2[i]

        for i in incident_id: #33
            model += x2[i] <= x1[i]

        model += pulp.lpSum(y[j] for j in base_id) == total_ambulances #34

        for j in base_id:
            model += y[j] <= max_abmulances[j]
        
        model.solve(pulp.PULP_CBC_CMD(msg = False))

        # -----------------------------
        # RESULTS
        # -----------------------------
        print("Status:", pulp.LpStatus[model.status])
        print("Objective (double-covered demand):", pulp.value(model.objective))

        print("\nAmbulance locations:")
        for j in base_id:
            if y[j].value() > 0:
                print(f"  Ambulance Base {j}: {y[j].value()} ambulances")

        print("\nDemand coverage:")
        for i in incident_id:
            print(f"  Demand {i}: x1={x1[i].value()}, x2={x2[i].value()}")


# Solve
import osmnx as ox
import pandas as pd
import numpy as np
import os
import json
import geopandas as gpd
from shapely import wkt

file_paths = {
    "ambulance_bases": "/home/thurein/ambo_allocate/integrate_map/ambulance_bases_data.csv",
    "grid": "/home/thurein/ambo_allocate/integrate_map/grid_data.csv",
    "graph": "/home/thurein/ambo_allocate/integrate_map/graph.graphml",
    # "hospitals": "/home/thurein/ambo_allocate/integrate_map/hospitals_data.csv"
}

# Check if files exist
for name, path in file_paths.items():
    if not os.path.exists(path):
        print(f"Error: {name} file not found at {path}")
    else:
        print(f"Found {name} file at {path}")

# Load files if they exist
if all(os.path.exists(path) for path in file_paths.values()):
    ambulance_bases_csv = pd.read_csv(file_paths["ambulance_bases"])
    grid_csv = pd.read_csv(file_paths["grid"])
    graph = ox.load_graphml(file_paths["graph"])
    # hospitals_csv = pd.read_csv(file_paths["hospitals"])
else:
    print("Some files are missing. Please check the paths.")

# Convert WKT back to geometry
for col in grid_csv.columns:
    if col in ['geometry']:
        grid_csv[col] = grid_csv[col].apply(wkt.loads)
# Convert back to GeoDataFrame
grid_gdf = gpd.GeoDataFrame(grid_csv, geometry='geometry', crs="EPSG:4326")
grid_gdf['index'] = grid_gdf.index
grid_gdf['Neighbor_nodes'] = grid_gdf['Neighbor_nodes'].apply(
    lambda x: json.loads(x) if pd.notna(x) and x != '' else []
)


# Convert WKT back to geometry
for col in ambulance_bases_csv.columns:
    if col in ['geometry']:
        ambulance_bases_csv[col] = ambulance_bases_csv[col].apply(wkt.loads)
# Convert back to GeoDataFrame
ambulance_bases_gdf = gpd.GeoDataFrame(ambulance_bases_csv, geometry='geometry', crs="EPSG:4326")
ambulance_bases_gdf['index'] = ambulance_bases_gdf.index
print("All data loaded from CSV and converted back to GeoDataFrames")

solver = DSM(graph,grid_gdf,ambulance_bases_gdf)

# Initialize simulation
max_ambulances= {
    0: 5, 
    1: 5,
    2: 5,
    3: 5,
    4: 5,
    5:5,
    6:5,
    7:5,
    8:5,
    9:5,
    10:5,
    11:5,
    12:5,
    13:5,
    14:5,
    15:5,
    16:5,
    17:5,
    18:5,
    19:5,
    20:5
}

solver.solve(total_ambulances=21,max_abmulances=max_ambulances)









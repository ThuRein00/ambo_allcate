# minimal_dqn.py
import gymnasium as gym
import torch
import DES_ambo_map_extended
from stable_baselines3 import PPO,DQN
import osmnx as ox
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import geopandas as gpd
from shapely import wkt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

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

# Initialize simulation
ambulance_allocation= {
    0: 0, 
    1: 0,
    2: 2,
    3: 0,
    4: 0,
    5:5,
    6:5,
    7:0,
    8:5,
    9:0,
    10:0,
    11:0,
    12:2,
    13:4,
    14:0,
    15:3,
    16:0,
    17:0,
    18:0,
    19:0,
    20:2
}

# Create environment using gym.make()
# env = gym.make(
#     "DES_ambo_map/DES_ambo_map-v1",
#     graph = graph,
#     grid = grid_gdf,
#     potential_base = ambulance_bases_gdf,
#     init_ambulances_per_base_dict=ambulance_allocation,
#     run_until=1440,
#     trace=True,
#     test = False
# )



# Load the trained model
model = PPO.load("/home/thurein/ambo_allocate/integrate_map/PPO_patient_count_with_demand_1", device = "cpu") 


# Evaluate the trained model
def model_evaluate(test_episodes,env,model=None):
    data_store_waiting_time = defaultdict(list)
    # data_store_incident_count = defaultdict(list)
    Reward = 0
    avg_time = defaultdict(float)
    for i in range (test_episodes):
        print(f"running episode {i}")
        # Reward = 0
        obs, _ = env.reset(seed=i)
        done = False

        while not done:
            if model: # if there is model, action is derived from model
                action, _ = model.predict(obs)
                
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated
            Reward = Reward+reward
        print(Reward)

        for incident_id in info.keys():
            data_store_waiting_time[incident_id] += list(info[incident_id])
    for incident_id in data_store_waiting_time.keys():
        avg_time[incident_id] = np.mean(np.array(data_store_waiting_time[incident_id]))

    return avg_time

# def waiting_time_boxplots(waiting_time_data = {}):
#     labels = sorted(list(waiting_time_data.keys()))
    
#     data = [waiting_time_data[label] for label in labels]

#     # Create figure
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Create boxplot with means and outliers
#     boxplot = ax.boxplot(
#         data,
#         patch_artist=True,
#         showmeans=True,       # show mean as a marker
#         meanline=False,       # mean as dot instead of line
#         flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'),
#         meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='orange')
#     )

#     # Set labels
#     ax.set_xticklabels(labels, rotation=30, ha='right')
#     ax.set_title("Avg waiting time by incident area")
#     ax.set_ylabel("Waiting Time (minutes)")

#     # Add colors for each box
#     colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightgray']
#     for patch, color in zip(boxplot['boxes'], colors * (len(data) // len(colors) + 1)):
#         patch.set_facecolor(color)

#     # Show grid
#     ax.yaxis.grid(True, linestyle='--', alpha=0.7)

#     plt.tight_layout()
#     plt.show()


# avg_time= model_evaluate(test_episodes=1
#                         ,env = env
#                         ,model=model)

# return to own base stats
env_test = gym.make(
    "DES_ambo_map/DES_ambo_map-v1",
    graph = graph,
    grid = grid_gdf,
    potential_base = ambulance_bases_gdf,
    init_ambulances_per_base_dict=ambulance_allocation,
    run_until=1440,
    trace=False,
    test = True
)

avg_time_return= model_evaluate(test_episodes=20
                                ,env = env_test,
                                model = None)

def compare_waiting_time_boxplots(waiting_time_model, waiting_time_baseline):
    labels = sorted(list(waiting_time_model.keys()))

    # Extract data
    data_model = [waiting_time_model[label] for label in labels]
    data_baseline = [waiting_time_baseline[label] for label in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Positions for boxplots
    positions_model = np.arange(len(labels)) * 2.0  # even positions
    positions_baseline = positions_model + 0.7      # offset baseline to the right

    # Boxplots
    bp_model = ax.boxplot(
        data_model,
        positions=positions_model,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none'),
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='orange')
    )

    bp_baseline = ax.boxplot(
        data_baseline,
        positions=positions_baseline,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none'),
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow')
    )

    # Colors
    for patch in bp_model['boxes']:
        patch.set_facecolor('lightblue')
    for patch in bp_baseline['boxes']:
        patch.set_facecolor('lightgreen')

    # X-axis labels in the middle between the two boxplots
    mid_positions = (positions_model + positions_baseline) / 2
    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Labels and title

    ax.set_title("Comparison of Avg Time by Incident Area")
    ax.set_ylabel("Time from incident to hospital (minutes)")
    ax.legend([bp_model["boxes"][0], bp_baseline["boxes"][0]],
                ["DQN", "Return-to-Own-Base"], loc="upper right")
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()



def compare_num_incident_barchart(incident_model, incident_baseline):
    labels = sorted(list(incident_model.keys()))

    # Calculate average incidents per area
    avg_model = [np.mean(incident_model[label]) for label in labels]
    avg_baseline = [np.mean(incident_baseline[label]) for label in labels]

    # Bar chart positions
    x = np.arange(len(labels))  # base positions
    bar_width = 0.35

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for model and baseline
    bars_model = ax.bar(x - bar_width/2, avg_model, width=bar_width, label="DQN", color="lightblue")
    bars_baseline = ax.bar(x + bar_width/2, avg_baseline, width=bar_width, label="Return-to-Own-Base", color="lightgreen")

    # Labels, title, and legend
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_title("Comparison of Avg Incident Count by Area")
    ax.set_ylabel("Average Incident Count per Simulation")
    ax.legend()

    # Add grid for clarity
    ax.yaxis.grid(True, linestyle="--")

    # Optionally add value labels on top of bars
    for bars in [bars_model, bars_baseline]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def compare_avg_time_barchart(incident_model, incident_baseline):
    labels = sorted(list(incident_model.keys()))

    # Calculate average incidents per area
    avg_model = [incident_model[label] for label in labels]
    avg_baseline = [incident_baseline[label] for label in labels]


    # Bar chart positions
    x = np.arange(len(labels))  # base positions
    bar_width = 0.35

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for model and baseline
    bars_model = ax.bar(x - bar_width/2, avg_model, width=bar_width, label="DQN", color="lightblue")
    bars_baseline = ax.bar(x + bar_width/2, avg_baseline, width=bar_width, label="Return-to-Own-Base", color="lightgreen")

    # Labels, title, and legend
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_title("Comparison of Avg time by Area")
    ax.set_ylabel("Average Time ")
    ax.legend()

    # Add grid for clarity
    ax.yaxis.grid(True, linestyle="--")

    # Optionally add value labels on top of bars
    for bars in [bars_model, bars_baseline]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# compare_waiting_time_boxplots(data_store_waiting_time,data_store_waiting_time_return)
# compare_num_incident_barchart(data_store_incident_count,data_store_incident_count_return)
# print(avg_time)
compare_avg_time_barchart(avg_time_return,avg_time_return)


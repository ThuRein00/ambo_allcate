# minimal_dqn.py
import gymnasium as gym
import torch
import DES_ambo_map_extended
from stable_baselines3 import PPO,DQN
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely import wkt
from torch import nn


import pandas as pd
import osmnx as ox
import os
import json
# Define file paths
file_paths = {
    "ambulance_bases": "/home/thurein/ambo_allocate/integrate_map/ambulance_bases_data.csv",
    "grid": "/home/thurein/ambo_allocate/integrate_map/grid_data.csv",
    "graph": "/home/thurein/ambo_allocate/integrate_map/graph.graphml",
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


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")


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
env = gym.make(
    "DES_ambo_map/DES_ambo_map-v1",
    graph = graph,
    grid = grid_gdf,
    potential_base = ambulance_bases_gdf,
    init_ambulances_per_base_dict=ambulance_allocation,
    run_until=1440,
    trace=False,
    test = False

)

policy_kwargs = dict(
    net_arch=[128, 128]   # larger policy network
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,       # good default, can lower to 1e-4 if unstable
    n_steps=2048,             # rollout length (increase if env is fast)
    batch_size=64,            # mini-batch size
    n_epochs=10,              # how many passes per update
    gamma=0.99,               # discount
    gae_lambda=0.95,          # GAE for advantage estimation
    clip_range=0.2,           # PPO clipping
    ent_coef=0.01,            # encourage exploration
    vf_coef=0.5,              # value function loss weight
    max_grad_norm=0.5,        # gradient clipping
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tb_logs_ppo/",
    device = "cpu"
)




# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=3e-4,        # similar LR as PPO
#     buffer_size=100000,        # replay buffer size (large enough to store experience)
#     learning_starts=1500,      # start learning after 1k steps
#     batch_size=64,             # same as PPO's batch size
#     tau=1.0,                   # target network update (soft update coefficient)
#     gamma=0.99,                # discount factor (same as PPO)
#     train_freq=4,              # train every 4 environment steps
#     gradient_steps=1,          # how many gradient steps per training step
#     target_update_interval=1500, # update target network every 1k steps
#     exploration_fraction=0.2,  # explore for 10% of total timesteps
#     exploration_final_eps=0.05, # final epsilon-greedy value
#     policy_kwargs=dict(net_arch=[64, 64]),  # 2-layer MLP, similar to PPO default
#     device=device,
#     tensorboard_log="./tb_logs1/"
# )
# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=3e-4,        # similar LR as PPO
#     buffer_size=100000,        # replay buffer size (large enough to store experience)
#     learning_starts=64*5,      # start learning after 1k steps
#     batch_size=64,             # same as PPO's batch size
#     tau=1.0,                   # target network update (soft update coefficient)
#     gamma=0.99,                # discount factor (same as PPO)
#     train_freq=4,              # train every 4 environment steps
#     gradient_steps=1,          # how many gradient steps per training step
#     target_update_interval=64*5, # update target network every 1k steps
#     exploration_fraction=0.2,  # explore for 10% of total timesteps
#     exploration_final_eps=0.05, # final epsilon-greedy value
#     policy_kwargs=dict(net_arch=[64, 64]),  # 2-layer MLP, similar to PPO default
#     device=device,
#     tensorboard_log="./tb_logs1/"
# )

# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=3e-4,        # similar LR as PPO
#     buffer_size=100000,        # replay buffer size (large enough to store experience)
#     learning_starts=1000,      # start learning after 1k steps
#     batch_size=64,             # same as PPO's batch size
#     tau=1.0,                   # target network update (soft update coefficient)
#     gamma=0.99,                # discount factor (same as PPO)
#     train_freq=4,              # train every 4 environment steps
#     gradient_steps=1,          # how many gradient steps per training step
#     target_update_interval=1000, # update target network every 1k steps
#     exploration_fraction=0.1,  # explore for 10% of total timesteps
#     exploration_final_eps=0.05, # final epsilon-greedy value
#     policy_kwargs=dict(net_arch=[64, 64,64]),  # 2-layer MLP, similar to PPO default
#     device=device,
#     tensorboard_log="./tb_logs1/"
# )

# from stable_baselines3 import DQN

# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=5e-5,            # ↓ smaller LR for stability
#     buffer_size=100_000,
#     learning_starts=5_000,
#     batch_size=128,                # ↑ smoother gradient
#     tau=1.0,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=1,
#     target_update_interval=4000,   # less frequent target updates
#     exploration_fraction=0.1,      # shorter exploration phase
#     exploration_final_eps=0.02,    # more exploitation after warmup
#     policy_kwargs=dict(net_arch=[64, 64]),
#     tensorboard_log="./tb_logs_dqn/"
# )



env.reset(seed=None)
# Train for 100,000 steps

# Try one single step manually
obs, _ = env.reset()
# done = True
# for _ in range (10): 
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated
    

model.learn(total_timesteps=100000)

# Save model
model.save("/home/thurein/ambo_allocate/integrate_map/PPO_patient_count_with_demand_1")

# Evaluate the trained model
# obs, _ = env.reset(seed=42)
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated 
#     print(f"  Action: {action} (Base {action})")
#     print(f"  Observation: {obs}")
#     print(f"  Reward: {reward}")

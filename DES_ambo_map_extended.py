import simpy
import numpy as np
from collections import defaultdict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import networkx as nx
import osmnx as ox

from shapely import wkt

import math
from geopy.distance import geodesic

class DES_ambo(gym.Env):
    metadata = {
                'render_modes': [],  #  no render modes supported
                'render_fps': 0      # Set to 0 since no rendering
                }
    def __init__(self ,graph = None, grid = None, potential_base = None ,init_ambulances_per_base_dict = None, run_until = 1440 ,trace=False, test = False):

        """
        Discrete Event Simulation for ambulance allocation
        :param env_config: Environment configuration object
        :param ambulances_per_base: Dictionary mapping base_id to ambulance count
        :param trace: Enable/disable event tracing
        """

        # Inherit from super class
        super(DES_ambo, self).__init__()

        self.trace_enabled = trace
        self.HOURS_PER_DAY = 1440
        self.AMBULANCE_SPEED = 1188 #m/min (70 min)

        # Graph Data
        self.graph = graph
        self.nodes, self.edges = ox.graph_to_gdfs(graph)
        self.grid = grid
        self.potential_base = potential_base

        # Initialize ambulances 
        self.init_ambulances_per_base_dict = init_ambulances_per_base_dict
        self.NUM_AMBULANCE_BASES = self.potential_base.shape[0]
        self.max_capacity = 20 # max num of ambulances at each base

        # For testing return to own base behavior
        self.test = test

        # How many minutes environment run
        self.run_until = run_until

        # Set action space
        # num_action = 4
        num_action = self.NUM_AMBULANCE_BASES
        self.action_space = spaces.Discrete(num_action) 

        # Set observation space
        self.observation_space = spaces.Box(
                                            low=0, 
                                            high=self.max_capacity, # max ambo at each base
                                            shape=(2*self.NUM_AMBULANCE_BASES,),  # 1D shape instead of 2D
                                            dtype=np.float32  # Standard 32-bit integer
                                            )
           

    def reset(self,seed = None , options = None ):
        #random number generator
        if not seed:
            seed = np.random.randint(0, 1000000)
        else:
            seed = seed 
        
        self.rng = np.random.default_rng(seed)

        # Statistics collection
        self.time_from_call_to_incident_dict = defaultdict(list)
        self.time_from_incident_to_hospital_dict = defaultdict(list)
        self.time_from_hospital_to_return_dict = defaultdict(list)
        self.time_from_call_to_hospital_dict = defaultdict(list)
        self.step_reward = {}
        self.start_time = {}
        self.reward = 0

        #new simpy env
        self.env = simpy.Environment()
        self.incident_counter = 0
        self.generate = False

        #for demand prediction with exponential smoothing
        self.now_incident_num = defaultdict(np.int32)
        self.next_incident_num = np.zeros(self.NUM_AMBULANCE_BASES)
        self.last_period = 0
        self.now_period = 1

        #received action initialization
        self.action_received = 0

        #initialize events
        self.relocation_complete = self.env.event()
        self.ambo_available = self.env.event()

        # store waiting incidents events
        self.waiting_incidents = []
        
        # Initialize Resources (ambulance bases)
        self.base_resources = {}
        for base_id, count in self.init_ambulances_per_base_dict.items():
            self.base_resources[base_id] = simpy.Container(
                self.env, 
                capacity=self.max_capacity, 
                init=count
            )

        self.assign_ambulances = [assign_ambulances for assign_ambulances in self.init_ambulances_per_base_dict.values()]
        observation = self._get_obs()
        info = self._get_info()
        
        return observation,info


    def trace(self, message):
        """Conditional tracing function"""
        if self.trace_enabled:
            print(f"{self.env.now:.2f}: {message}")

    def Eu_distance_nodes(self,node1_id, node2_id):
        """Calculate Eucledian Distance"""
        node1_data = self.graph.nodes[node1_id]
        node2_data = self.graph.nodes[node2_id]
        dx = node2_data['x'] - node1_data['x']
        dy = node2_data['y'] - node1_data['y']
        return (dx**2 + dy**2)**0.5

    
    def incident_generator(self):
        """Generate incidents at each location according to Poisson process"""
        # Generate incidents at each grid with mean not equal to 0
        for row in self.grid.itertuples():
            if row.mean_rate !=0:
                yield self.env.timeout(0)
                incident_id = row.Index
                mean_interarrival = self.HOURS_PER_DAY/ row.mean_rate 
                self.env.process(self._incident_process(incident_id, mean_interarrival))
    
    def _incident_process(self, incident_id, mean_interarrival):
        """Process for generating incidents at a specific location"""
        while True:
            # Exponential interarrival times
            inter_arrival = self.rng.exponential(mean_interarrival) 
            yield self.env.timeout(inter_arrival)
            
            # Create new incident
            incident_no = self.incident_counter
            self.incident_counter += 1
            incident_polygon = self.grid.loc[incident_id,'geometry']

            # Filter nodes that fall inside the polygon
            nodes_in_poly = self.nodes[self.nodes.within(incident_polygon)]

            # Pick a random node from those
            if not nodes_in_poly.empty:
                random_node = nodes_in_poly.sample(1)   # pandas/GeoPandas sample
                incident_node_id = random_node.index[0] 
            else:
                break

            # Find nearest base to assign incident count
            min_distance = float('inf')
            for base_id,base_node_id in zip(self.potential_base['index'],self.potential_base['nearest_node_id']):
                distance = self.Eu_distance_nodes(incident_node_id,base_node_id)
            
                if distance < min_distance:
                    min_distance = distance
                    nearest_base_id = base_id

            self.now_incident_num[nearest_base_id] = self.now_incident_num[nearest_base_id] +1

            

        

            #find free ambulance
            self.env.process(self.find_free_ambo(incident_no, incident_id,incident_node_id))
            
    def find_free_ambo(self, incident_no, incident_id,incident_node_id):
        self.start_time[incident_no] = self.env.now

        # Find bases with available ambulances
        available_bases = []
        for base_id, resource in self.base_resources.items():
            if resource.level > 0:
                available_bases.append(base_id)

        
        # Eucledian minimun
        if available_bases:

            # Find closest base
            min_distance = float('inf')
            nearest_base = None
            
            for base_id in available_bases:
                base_node_id = self.potential_base.loc[base_id,'nearest_node_id']

                distance = self.Eu_distance_nodes(base_node_id,incident_node_id)
            
                if distance < min_distance:
                    min_distance = distance
                    nearest_base = base_id
                    nearest_base_node_id = base_node_id

            # Use A* algorithm to calculate actual distance
            if nx.has_path(self.graph, source=nearest_base_node_id, target=incident_node_id):
                actual_min_distance = nx.astar_path_length(
                    self.graph, source=nearest_base_node_id, target=incident_node_id, 
                    heuristic= self.Eu_distance_nodes, weight='length'
                )
            else:
                #if there is no actual use Eucledian estimation
                actual_min_distance = min_distance * 1.3

            # Dispatch ambulance
            self.trace(f"Incident {incident_no} at location {incident_id} with Nearest available base: {nearest_base} (distance: {min_distance:.2f})")
            self.env.process(self.ambulance_dispatch(
                incident_no, 
                incident_id,
                incident_node_id,
                nearest_base, 
                actual_min_distance
            ))


        # If there is no ambulance base available at the time of call, wait for the first available ambulance
        else:
            # Create a new event for this specific incident
            ambo_available = self.env.event()
            
            # Store the event in a dictionary with incident ID as key
            self.waiting_incidents.append(ambo_available)
            
            # self.trace(f"Incident {incident_no} at location {incident_id} is waiting for an ambulance")
            
            # Wait for the event to be triggered
            yield ambo_available
            
            # Find which base now has an ambulance
            available_bases = []
            for base_id, resource in self.base_resources.items():
                if resource.level > 0:
                    available_bases.append(base_id)
                    self.trace(f"Base {base_id} becomes available for incident {incident_no}")

            # Eucledian minimun
            min_distance = float('inf')
            nearest_base = None
            
            for base_id in available_bases:
                base_node_id = self.potential_base.loc[base_id,'nearest_node_id']

                distance = self.Eu_distance_nodes(base_node_id,incident_node_id)
            
                if distance < min_distance:
                    min_distance = distance
                    nearest_base = base_id
                    nearest_base_node_id = base_node_id

                    
            self.trace(f"Incident {incident_no} at location {incident_id} with Nearest available base: {nearest_base} (distance: {min_distance:.2f})")

            # Use A* algorithm
            if nx.has_path(self.graph, source=nearest_base_node_id, target=incident_node_id):
                actual_min_distance = nx.astar_path_length(
                    self.graph,source= nearest_base_node_id, target=incident_node_id, 
                    heuristic= self.Eu_distance_nodes, weight='length'
                )
            else:
                actual_min_distance = min_distance * 1.3

            # Dispatch ambulance
            self.env.process(self.ambulance_dispatch(
                incident_no, 
                incident_id,
                incident_node_id,
                nearest_base, 
                actual_min_distance
            ))
            

    def ambulance_dispatch(self, incident_no, incident_id, incident_node_id,base_id, distance):
        """Process for ambulance dispatch and transport to incident"""
        base_resource = self.base_resources[base_id]
        
        # Request ambulance
        yield base_resource.get(1)

        # change ambulance number at base
        self.assign_ambulances[base_id] -= 1
        
        # Travel to incident
        travel_time = distance / self.AMBULANCE_SPEED  
        yield self.env.timeout(travel_time+1)
        
        # Record time from call to incident
        time_from_call_to_incident = self.env.now - self.start_time.get(incident_no)
    
        self.trace(f"Ambulance arrived at incident: {incident_no} form base: {base_id}(response time: {time_from_call_to_incident:.2f} min)")
        
        # Find nearest hospital
        min_distance = float('inf')

        for hospital_node_id in self.grid.loc[incident_id, 'Neighbor_nodes']:
            distance = self.Eu_distance_nodes(node1_id=incident_node_id,node2_id=hospital_node_id)

            if distance < min_distance:
                min_distance = distance
                nearest_hospital_node_id = hospital_node_id

        # Use A* algorithm to get actual distance 
        if nx.has_path(self.graph, source=incident_node_id, target=nearest_hospital_node_id):
            actual_min_distance = nx.astar_path_length(
                self.graph, source=incident_node_id, target=nearest_hospital_node_id, 
                heuristic= self.Eu_distance_nodes, weight='length'
            )
        else:
            actual_min_distance = min_distance * 1.3

        # Transport to hospital
        self.env.process(self.transport_to_hospital(
            incident_no,
            incident_id,
            base_id,
            nearest_hospital_node_id,
            actual_min_distance,
            time_from_call_to_incident
        ))
    
    def transport_to_hospital(self, incident_no, incident_id,base_id, hospital_node_id ,distance,time_from_call_to_incident):
        """Process for transporting patient to hospital"""
        # Travel to hospital
        travel_time = (distance / self.AMBULANCE_SPEED) + (self.rng.random())
        yield self.env.timeout(travel_time+2)

        # Record time from incident to hospital
        time_from_incident_to_hospital = (self.env.now - self.start_time.get(incident_no)) - time_from_call_to_incident

        self.trace(f"Patient from incident {incident_no} at incident id {incident_id} arrived at hospital {hospital_node_id} with ambulance from base {base_id} Take time {time_from_call_to_incident+time_from_incident_to_hospital:.2f}")

        self.env.process(self.relocate_base(incident_no, 
                                            incident_id,
                                            base_id, 
                                            hospital_node_id,
                                            time_from_call_to_incident,
                                            time_from_incident_to_hospital))
    
    def return_to_own_base(self,hospital_node_id,base_id):
        relocate_base_node_id = self.potential_base.loc[base_id,'nearest_node_id']

        if nx.has_path(self.graph, source=hospital_node_id, target=relocate_base_node_id):
            distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=relocate_base_node_id, weight='length')
        else:
            #if there is no connected path, use estimate distance
            distance = self.Eu_distance_nodes(node1_id=hospital_node_id,node2_id=relocate_base_node_id) * 1.3
            print("here")

        self.trace(f"ambo go back to own base {base_id}")
        
        return distance,base_id

    def back_to_nearest_base(self,hospital_node_id):
        min_distance = float('inf')
        nearest_base_node_id = None
            
        for base_id,base_node_id in zip(self.potential_base['index'],self.potential_base['nearest_node_id']):
            distance = self.Eu_distance_nodes(hospital_node_id,base_node_id)
            
            if distance < min_distance:
                min_distance = distance
                nearest_base_node_id = base_node_id
                nearest_base_id = base_id
        
        if nx.has_path(self.graph, source=hospital_node_id, target=nearest_base_node_id):
            distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=nearest_base_node_id, weight='length')
        else:
            #if there is no connected path, use estimate distance
            distance = min_distance * 1.3
            print("here")

        self.trace(f"ambo go back to nearest base {nearest_base_id}")

        return distance,nearest_base_id
        
    def back_to_base_with_less_ambo(self,hospital_node_id):
        num_ambulances = np.array(self.assign_ambulances)
        min_value = np.min(num_ambulances)                     # find smallest number
        lowest_num_bases = np.where(num_ambulances == min_value)[0]  # get ALL indices
        chosen_base_id = np.random.choice(lowest_num_bases)
        chosen_base_node_id = self.potential_base['nearest_node_id'][chosen_base_id]
        
        if nx.has_path(self.graph, source=hospital_node_id, target=chosen_base_node_id):
            distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=chosen_base_node_id, weight='length')
        else:
            #if there is no connected path, use estimate distance
            distance = self.Eu_distance_nodes(node1_id=hospital_node_id,node2_id=chosen_base_node_id) * 1.3
            print("here")

        self.trace(f"ambo go back to base {chosen_base_id} with less ambo")
        return distance,chosen_base_id

    def back_to_highest_expected_demand (self,hospital_node_id):        
        expected_rate = np.array(self.next_incident_num)
        max_value = np.max(expected_rate)

        highest_rate_incident_area = np.where(expected_rate == max_value)[0]  # extract array
        chosen_incident_area = np.random.choice(highest_rate_incident_area)

        chosen_incident_centroid = self.grid['geometry'][chosen_incident_area].centroid
        x, y = chosen_incident_centroid.x, chosen_incident_centroid.y

        # find nearest base from chosen incident area (centroid)
        min_distance = float('inf')
        nearest_base_node_id = None
            
        for base_id,base_node_id in zip(self.potential_base['index'],self.potential_base['nearest_node_id']):
            node_data = self.graph.nodes[base_node_id]
            dx = node_data['x'] - x
            dy = node_data['y'] - y
            distance = (dx**2 + dy**2)**0.5

            if distance < min_distance:
                min_distance = distance
                nearest_base_node_id = base_node_id
                nearest_base_id = base_id

        # find actual distance from hospital to base
        if nx.has_path(self.graph, source=hospital_node_id, target=nearest_base_node_id):
            distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=nearest_base_node_id, weight='length')
        else:
            #if there is no connected path, use estimate distance
            distance = self.Eu_distance_nodes(node1_id=hospital_node_id,node2_id=nearest_base_node_id)  * 1.3

        self.trace("ambo go back to base highest expected demand")
        
        return distance,nearest_base_id

    def relocate_base(self, incident_no, incident_id,base_id, hospital_node_id, time_from_call_to_incident, time_from_incident_to_hospital):
        """Process for returning ambulance to base according to action"""
        if not self.test:
            relocate_base_id = self.action_received
            relocate_base_node_id = self.potential_base['nearest_node_id'][relocate_base_id]

            # Calculate return distance
            if nx.has_path(self.graph, source=hospital_node_id, target=relocate_base_node_id):
                distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=relocate_base_node_id, weight='length')
            else:
                #if there is no connected path, use estimate distance
                distance = self.Eu_distance_nodes(node1_id=hospital_node_id,node2_id=relocate_base_node_id) * 1.3

        else:
            if self.action_received == 0:
                # print("return to own base")
                distance,relocate_base_id = self.return_to_own_base(hospital_node_id,base_id)

            elif self.action_received == 1:
                # print("return to nearest base")
                distance,relocate_base_id = self.back_to_nearest_base(hospital_node_id)

            elif self.action_received == 2:
                # print("return to base with less ambo")
                distance,relocate_base_id = self.back_to_base_with_less_ambo(hospital_node_id)
                
            else:
                # print("return to highest expected demand")
                distance,relocate_base_id = self.back_to_highest_expected_demand(hospital_node_id)

        
        # Travel back to base
        travel_time = (distance / self.AMBULANCE_SPEED) + (self.rng.random())
        yield self.env.timeout(travel_time+3)

        #collect stats
        self.time_from_call_to_incident_dict[incident_id].append(time_from_call_to_incident)
        # self.time_from_incident_to_hospital_dict[incident_id].append(time_from_incident_to_hospital)
        # self.time_from_call_to_hospital_dict[incident_id].append(time_from_call_to_incident+time_from_incident_to_hospital)

        # Record time from hospital to return
        time_from_hospital_to_return = (self.env.now - self.start_time.get(incident_no))\
        - time_from_call_to_incident - time_from_incident_to_hospital
        del self.start_time[incident_no] #retrieve and delete value
        
        if (time_from_call_to_incident) <= 7:
            self.step_reward[incident_no] = 1
        else:
            self.step_reward[incident_no] = 0

        # self.step_reward[incident_no] = 3*time_from_call_to_incident + time_from_incident_to_hospital + 0.5*time_from_hospital_to_return 
        # self.step_reward[incident_no] = (time_from_call_to_incident)**2 + 0*time_from_incident_to_hospital + 0*time_from_hospital_to_return 
        # self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital + 0.5*time_from_hospital_to_return #Good
        # self.step_reward[incident_no] = time_from_call_to_incident #(not good)
        # self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital # good)
        # self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital + time_from_hospital_to_return # noot good
        # print(f"step_reward is {time_from_call_to_incident + time_from_incident_to_hospital}")
        # calculate reward 

        self.reward = self.step_reward[incident_no]
        del self.step_reward[incident_no]

        
        # Return ambulance to base
        base_resource = self.base_resources[relocate_base_id]
        yield base_resource.put(1)

        # Update ambulance counts
        if not self.relocation_complete.triggered:
            self.relocation_complete.succeed()
            self.assign_ambulances[relocate_base_id] += 1
        

        # triggered this only if there is waiting incidents
        if self.waiting_incidents:
            event_to_trig = self.waiting_incidents.pop(0)
            if not event_to_trig.triggered:
                event_to_trig.succeed()


    def _get_info(self):
        """Calculate statistics."""
        return self.time_from_call_to_incident_dict
        
    
    def _get_obs(self):
        num_ambulances_at_each_base = self.assign_ambulances
        self.trace(f"total avalilable ambo = {np.sum(num_ambulances_at_each_base)}")
        expected_patient_at_each_base = self.next_incident_num
        observation = np.array(list(num_ambulances_at_each_base)+ list(expected_patient_at_each_base),dtype=np.float32)
        return observation
    
    def step(self, action):
        self.action_received = action
        self.trace(f"Action received: {action}")

        self.relocation_complete = self.env.event()

        # Start the incident generator if not already running
        if not self.generate: # generate only one time
            self.incident_process = self.env.process(self.incident_generator())
            self.generate = True
    
        # run 
        self.env.run(until=self.relocation_complete)
        
        # at every hour reset now_incident_num and update next incident num
        period = int(self.env.now/240) +1 # four hours is one period
    
        self.now_period = period

        if self.now_period != self.last_period:
            # predict next period incident with exponential smoothing
            alpha = 0.2
            # print(f"now period is {self.now_period}")
            true_D = np.zeros(self.NUM_AMBULANCE_BASES)

            for base_id in self.potential_base['index']:
                if base_id in self.now_incident_num.keys():
                    true_D[base_id] = self.now_incident_num[base_id]
                else:
                    true_D[base_id] = 0
            # print(f"true demand is {true_D}")

            for base_id in self.potential_base['index']:
                if base_id in self.now_incident_num.keys():
                    self.next_incident_num[base_id] = ((alpha * self.now_incident_num[base_id] + (1-alpha)*self.next_incident_num[base_id]))/(self.now_period-self.last_period)
                else:
                    self.next_incident_num[base_id] = ((alpha * 0+ (1-alpha)*self.next_incident_num[base_id]))/(self.now_period-self.last_period)
            # print(f"Expected demand in next period is {self.next_incident_num}")
            # reset incident count
            self.now_incident_num = defaultdict(np.int32)
            self.last_period = self.now_period
            
    

        # Get observation
        observation = self._get_obs()
        self.trace(f"Now state is {observation}")
        # Calculate reward
        reward = self.reward

        terminated = self.env.now >= self.run_until
   
        truncated = False #never fail or later add something

        # Get information of each day
        if terminated: 
            # info = self._get_info()
            info = self._get_info() #no info is passed during training
        else:
            info = {}
    
        return observation, reward, terminated, truncated, info
        
    
register(
    id="DES_ambo_map/DES_ambo_map-v1",
    entry_point=DES_ambo,  # The class defined in this file
    kwargs={
        'graph': None,
        'grid' : None,
        'potential_base' : None,
        'init_ambulances_per_base_dict': None,
        'run_until': 1440,
        'trace': False,
        'test' : False
    }
)






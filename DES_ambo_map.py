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
    def __init__(self ,graph = None, grid = None, potential_base = None ,init_ambulances_per_base_dict = None, run_until = 1440 ,trace=False, return_own_base = False):
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
        self.AMBULANCE_SPEED = 1833 #m/min

        # Environment configuration
        self.graph = graph
        self.nodes, self.edges = ox.graph_to_gdfs(graph)
        self.grid = grid
        self.potential_base = potential_base
        self.init_ambulances_per_base_dict = init_ambulances_per_base_dict
        self.return_own_base = return_own_base

       

        # max ambulances fit in each base
        self.max_capacity = 20

        self.run_until = run_until
        self.NUM_AMBULANCE_BASES = self.potential_base.shape[0]
        
        # Set action space (as a choice from dispatch points)
        self.action_space = spaces.Discrete(self.NUM_AMBULANCE_BASES) # need to map 0,1,2 to A,B,C   

        # Set observation space: number of ambulances currently assigned to each base
        self.observation_space = spaces.Box(
                                            low=0, 
                                            high=self.max_capacity,  # Note: removed -1 to include max capacity
                                            shape=(self.NUM_AMBULANCE_BASES,),  # 1D shape instead of 2D
                                            dtype=np.int32  # Standard 32-bit integer
                                            )
           
        # Validate ambulance configuration
        # self.validate_ambulance_config(self.init_ambulances_per_base_dict)
        

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
        self.step_reward = {}
        self.reward = 0
        self.start_time = {}

        #new simpy env
        self.env = simpy.Environment()
        self.incident_counter = 0
        self.generate = False

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
                capacity=self.max_capacity, # max capacity is preset to 20
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

    # def Eu_distance_nodes (self,node1_id,node2_id):
    #     node1_data = self.graph.nodes[node1_id]
    #     node2_data = self.graph.nodes[node2_id]
    #     coord1= (node1_data['y'],node1_data['x']) #lat/lon
    #     coord2 = (node2_data['y'],node2_data['x'])

    #     return geodesic(coord1, coord2).meters
    def Eu_distance_nodes(self,node1_id, node2_id):
        node1_data = self.graph.nodes[node1_id]
        node2_data = self.graph.nodes[node2_id]
        dx = node2_data['x'] - node1_data['x']
        dy = node2_data['y'] - node1_data['y']
        return (dx**2 + dy**2)**0.5

    
    def incident_generator(self):
        """Generate incidents at each location according to Poisson process"""

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
            # inter_arrival = mean_interarrival
            yield self.env.timeout(inter_arrival)
            
            # Create new incident
            incident_no = self.incident_counter
            self.incident_counter += 1

            #incident node
            # # Convert only if it’s a string
            # if isinstance(self.grid.loc[incident_id, 'geometry'], str):
            #     self.grid['geometry_wgs84'] = self.grid['geometry_wgs84'].apply(wkt.loads)

            incident_polygon = self.grid.loc[incident_id,'geometry']


            # Filter nodes that fall inside the polygon
            nodes_in_poly = self.nodes[self.nodes.within(incident_polygon)]


            # Pick a random node from those
            if not nodes_in_poly.empty:
                random_node = nodes_in_poly.sample(1)   # pandas/GeoPandas sample
                incident_node_id = random_node.index[0] 
            else:
                break

            #find free ambulance
            self.env.process(self.find_free_ambo(incident_no, incident_id,incident_node_id))
            
    def find_free_ambo(self, incident_no, incident_id,incident_node_id):
        self.start_time[incident_no] = self.env.now
       
        # Find bases with available ambulances
        available_bases = []
        for base_id, resource in self.base_resources.items():
            if resource.level > 0:
                available_bases.append(base_id)

        # # actual minimun
        # if available_bases:
        #     # Find closest base
        #     min_distance = float('inf')
        #     nearest_base = None
            
        #     for base_id in available_bases:
        #         base_node_id = self.potential_base.loc[base_id,'nearest_node_id']

        #         if nx.has_path(self.graph, source=base_node_id, target=incident_node_id):
        #             distance = nx.shortest_path_length(self.graph, source=base_node_id, target=incident_node_id, weight='length')
            
        #         if distance < min_distance:
        #             min_distance = distance
        #             nearest_base = base_id

        
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

            # if nx.has_path(self.graph, source=nearest_base_node_id, target=incident_node_id):
            #     actual_min_distance = nx.shortest_path_length(self.graph, source=nearest_base_node_id, target=incident_node_id, weight='length')
            # else:
            #     actual_min_distance = min_distance *1.3 #use estimate if there is no connection between nodes
                    # Use A* algorithm
        # Use A* algorithm
            if nx.has_path(self.graph, source=nearest_base_node_id, target=incident_node_id):
                actual_min_distance = nx.astar_path_length(
                    self.graph, source=nearest_base_node_id, target=incident_node_id, 
                    heuristic= self.Eu_distance_nodes, weight='length'
                )
            else:
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


        # If there is no ambulance base available at the time of call, 
        # wait for the first available ambulance
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
            
            # # Find the nearest base from the available ones (actual minimun)
            # min_distance = float('inf')
            # nearest_base = None
            
            # for base_id in available_bases:
            #     base_node_id = self.potential_base.loc[base_id,'nearest_node_id']
                                
            #     if nx.has_path(self.graph, source=base_node_id, target=incident_node_id):
            #         distance = nx.shortest_path_length(self.graph, source=base_node_id, target=incident_node_id, weight='length')
            #         if distance < min_distance:
            #             min_distance = distance
            #             nearest_base = base_id


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
            #calculate actual min distance

            # if nx.has_path(self.graph, source=nearest_base_node_id, target=incident_node_id):
            #     actual_min_distance = nx.shortest_path_length(self.graph, source=nearest_base_node_id, target=incident_node_id, weight='length')
            # else:
            #     actual_min_distance = min_distance *1.3 #use estimate if there is no connection between nodes

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
        yield self.env.timeout(travel_time)
        
        # Record time from call to incident
        time_from_call_to_incident = self.env.now - self.start_time.get(incident_no)
        # del self.start_time[incident_no] #retrieve and delete value

        #collect stats
        self.time_from_call_to_incident_dict[incident_id].append(time_from_call_to_incident)
        # self.step_reward[incident_no] = response_time_1

        self.trace(f"Ambulance arrived at incident: {incident_no} form base: {base_id}(response time: {time_from_call_to_incident:.2f} min)")
        
        # Find nearest hospital (hospital and bases are the same in this case)

        min_distance = float('inf')
        # nearest_hospital = None

        # neighbor_poly = self.grid['geometry'][self.grid[f'neighbor_to_{incident_id}']]
        # multi_poly = neighbor_poly.union_all()  # Merge all 
        # hospital_in_poly = self.hospitals[['index','nearest_node_id']][self.hospitals.geometry.within(multi_poly)]

        # neighbor_poly = self.grid['geometry'][self.grid[f'neighbor_to_{incident_id}']]
        # multi_poly = neighbor_poly.unary_union  # Use unary_union instead of union_all

        # Filter hospitals within the polygon and get their node IDs
        # hospital_in_poly = self.hospitals.loc[
        #     self.hospitals.geometry.within(multi_poly),
        #     ['index', 'nearest_node_id']
        # ]

        for hospital_node_id in self.grid.loc[incident_id, 'Neighbor_nodes']:
            # if nx.has_path(self.graph, source=incident_node_id, target=hospital_node_id):
            #     distance = nx.shortest_path_length(self.graph, source=incident_node_id, target=hospital_node_id, weight='length')
            distance = self.Eu_distance_nodes(node1_id=incident_node_id,node2_id=hospital_node_id)

            if distance < min_distance:
                min_distance = distance
                nearest_hospital_node_id = hospital_node_id

        # if nx.has_path(self.graph, source=incident_node_id, target=nearest_hospital_node_id):
        #     actual_min_distance = nx.shortest_path_length(self.graph, source=incident_node_id, target=nearest_hospital_node_id, weight='length')
        # else:
        #     actual_min_distance = min_distance *1.3 #use estimate if there is no actual connected nodes

        # Use A* algorithm
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
        travel_time = distance / self.AMBULANCE_SPEED + (self.rng.random())
        yield self.env.timeout(travel_time)

        # Record time from incident to hospital
        time_from_incident_to_hospital = (self.env.now - self.start_time.get(incident_no)) - time_from_call_to_incident
        # del self.start_time[incident_no] #retrieve and delete value

        #collect stats
        self.time_from_incident_to_hospital_dict[incident_id].append(time_from_incident_to_hospital)
        # self.step_reward[incident_no] = response_time_1

        self.trace(f"Patient from incident {incident_no} at incident id {incident_id} arrived at hospital {hospital_node_id} with ambulance from base {base_id}")

        self.env.process(self.relocate_base(incident_no, incident_id,base_id, hospital_node_id,self.relocate_base_id,
                                            time_from_call_to_incident,time_from_incident_to_hospital))
    

    
    def relocate_base(self, incident_no, incident_id,base_id, hospital_node_id,relocate_base_id, time_from_call_to_incident, time_from_incident_to_hospital):
        """Process for returning ambulance to base according to action"""
        # Calculate return distance
        if not self.return_own_base:
            relocate_base_node_id = self.potential_base.loc[relocate_base_id,'nearest_node_id']
        else:
            relocate_base_node_id = self.potential_base.loc[base_id,'nearest_node_id']
        
        if nx.has_path(self.graph, source=hospital_node_id, target=relocate_base_node_id):
            distance = nx.shortest_path_length(self.graph, source=hospital_node_id, target=relocate_base_node_id, weight='length')
        else:
            #if there is no connected path, use estimate distance
            distance = self.Eu_distance_nodes(node1_id=hospital_node_id,node2_id=relocate_base_node_id) * 1.3

        
        # Travel back to base
        travel_time = distance / self.AMBULANCE_SPEED + (self.rng.random())
        yield self.env.timeout(travel_time)

        # Record time from hospital to return
        time_from_hospital_to_return = (self.env.now - self.start_time.get(incident_no))\
        - time_from_call_to_incident - time_from_incident_to_hospital
        del self.start_time[incident_no] #retrieve and delete value

        # self.step_reward[incident_no] = 3*time_from_call_to_incident + time_from_incident_to_hospital + 0.5*time_from_hospital_to_return 
        # self.step_reward[incident_no] = (time_from_call_to_incident)**2 + 0*time_from_incident_to_hospital + 0*time_from_hospital_to_return 
        # self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital + 0.5*time_from_hospital_to_return #Good
        # self.step_reward[incident_no] = time_from_call_to_incident #(not good)
        self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital # good)
        # self.step_reward[incident_no] = time_from_call_to_incident + time_from_incident_to_hospital + time_from_hospital_to_return # noot good
        

        
        # Return ambulance to base
        base_resource = self.base_resources[relocate_base_id]
        yield base_resource.put(1)
    
        # Update ambulance counts
        if not self.relocation_complete.triggered:
            self.relocation_complete.succeed()
            self.assign_ambulances[relocate_base_id] += 1
            

        # calculate reward 
        self.reward = self.step_reward[incident_no]
        del self.step_reward[incident_no]

        # triggered this only if there is waiting incidents
        if self.waiting_incidents:
            event_to_trig = self.waiting_incidents.pop(0)
            if not event_to_trig.triggered:
                event_to_trig.succeed()


    def _get_info(self):
        """Calculate statistics."""
        # stats_time_call_to_incident = {}
        # for incident_id, times in self.time_from_call_to_incident_dict.items():
        #     if times:
        #         stats_time_call_to_incident[incident_id] = {
        #                 'mean': np.mean(times),
        #                 'min': min(times),
        #                 'max': max(times),
        #                 'count': len(times)
        #             }
                
                
        # stats_time_incident_to_hospital = {}
        # for incident_id, times in self.time_from_incident_to_hospital_dict.items():
        #     if times:
        #         stats_time_incident_to_hospital[incident_id] = {
        #                 'mean': np.mean(times),
        #                 'min': min(times),
        #                 'max': max(times),
        #                 'count': len(times)
        #             }

        # return {0:stats_time_call_to_incident,1:stats_time_incident_to_hospital}
                
        time_from_call_to_hospital_dict = defaultdict(list)
        for incident_id, times in self.time_from_call_to_incident_dict.items():
            if len(times)>0 :
                for _ in range(len(self.time_from_call_to_incident_dict[incident_id]) - len(self.time_from_incident_to_hospital_dict[incident_id])):
                    self.time_from_call_to_incident_dict[incident_id].pop() # sometimes at the day end some ambulances do not return to the base yet
                time_from_call_to_hospital_dict[incident_id] = np.array(self.time_from_call_to_incident_dict[incident_id]) \
                                                        + np.array(self.time_from_incident_to_hospital_dict[incident_id]) 

                
        stats_time_call_to_hospital = {}
        for incident_id, times in time_from_call_to_hospital_dict.items():
            if len(times)>0 :
                stats_time_call_to_hospital[incident_id] = {
                            'mean': np.mean(times),
                            'min': min(times),
                            'max': max(times),
                            'count': len(times)
                        }
        
        return stats_time_call_to_hospital
        
    
    def _get_obs(self):
        observation = np.array(self.assign_ambulances,dtype=np.int32 )
        return observation
    
    def step(self, action):
        self.relocate_base_id = action
        self.trace(f"Action received: {action} (base {self.relocate_base_id})")

        self.relocation_complete = self.env.event()

        # Start the incident generator if not already running
        if not self.generate: # generate only one time

            self.incident_process = self.env.process(self.incident_generator())
            self.generate = True
    
        # Create a timeout event to prevent infinite waiting
        # timeout_event = self.env.timeout(self.run_until - self.env.now)
    
        # Run until either relocation is complete or we timeout
        # result = self.env.run(until=self.env.any_of([self.relocation_complete, timeout_event]))
        self.env.run(until=self.relocation_complete)

    
        # # Check which event caused the simulation to stop
        # if self.relocation_complete in result:
        #     self.trace("Step completed due to relocation finish")
        # else:
        #     self.trace("Step completed due to timeout")

        # Get observation
        observation = self._get_obs()

        # Calculate reward
        reward = -self.reward
    
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
    id="DES_ambo_map/DES_ambo_map-v0",
    entry_point=DES_ambo,  # The class defined in this file
    kwargs={
        'graph': None,
        'grid' : None,
        'potential_base' : None,
        'init_ambulances_per_base_dict': None,
        'run_until': 1440,
        'trace': False
    }
)




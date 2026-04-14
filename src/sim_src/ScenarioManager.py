import math
import pandas as pd
import numpy as np
import json

from EntityManager import EntityManager
from EventManager import EventManager
class ScenarioManager():
    def __init__(self, configs, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.scenario = {}

        # 1. Create entity information
        # Exclude metadata such as departure_time; extract only actual entities
        entity_keys = [k for k in configs['entity_info'].keys()
                       if k in ["patient", "hospital", "ambulance", "uav"]]
        self.en_manager = EntityManager(entity_keys)

        for en_name, raw_prop in configs['entity_info'].items():
            if en_name == "patient":
                reg_prop = self.setup_patient(raw_prop)
            elif en_name == "hospital":
                reg_prop = self.setup_hospital(raw_prop)
            elif en_name == "ambulance":
                reg_prop = self.setup_ambulance(raw_prop)
            elif en_name == "uav":
                reg_prop = self.setup_uav(raw_prop)
            elif en_name == "departure_time":
                # Metadata: not used in simulation (used during scenario generation)
                continue
            else:
                raise NotImplementedError(f"{en_name} is not yet implemented as an entity.")
            self.en_manager.en_register(en_name, reg_prop)
        self.scenario['EntityManager'] = self.en_manager

        # 2. Create event information
        with open(configs['event_info_path'], "r") as f:
            ev_info = json.load(f)
        self.ev_manager = EventManager(ev_info, self.en_manager, rng=self.rng)
        self.scenario['EventManager'] = self.ev_manager

    def set_seed(self, rng):
        self.rng = rng
        self.ev_manager.set_seed(rng)
    def setup_patient(self, cfg_patient):
        reg_prop ={}
        reg_prop['incident_size'] = cfg_patient['incident_size']
        incident_loc = (cfg_patient['latitude'], cfg_patient['longitude'])
        incident_type = cfg_patient['incident_type']
        try:
            patient_info = pd.read_csv(cfg_patient['info_path'])
            required_cols = [
                'ratio',
                'rescue_param_alpha',
                'rescue_param_beta',
                'treat_tier3',
                'treat_tier2',
                'treat_tier3_mean',
                'treat_tier2_mean',
            ]
            missing_cols = [col for col in required_cols if col not in patient_info.columns]
            if missing_cols:
                raise KeyError(f"patient_info.csv missing required columns: {missing_cols}")
            if incident_type is not None:
                raise NotImplementedError("Incident type information is not yet implemented.")
            assert math.isclose(patient_info['ratio'].sum(), 1.0), "Patient ratios must sum to 1."
            reg_prop['patient_info'] = patient_info
            # p_info_dict = patient_info.set_index("type").to_dict(orient="index")
            # reg_prop.update({'Red': p_info_dict['Red'],
            #                  'Yellow': p_info_dict['Yellow'],
            #                  'Green': p_info_dict['Green'],
            #                  'Black': p_info_dict['Black']})
        except FileNotFoundError:
            print("Patient data was not provided.")
        return reg_prop

    def get_lognormal_param(self, m):
        v = np.power(m * 0.4, 2)
        mean_logn = np.zeros_like(m, dtype=float)
        mask = m > 0 # Keep elements with value 0 as 0
        mean_logn[mask] = np.log(m[mask] / np.sqrt(1 + v[mask] / np.power(m[mask], 2)))
        std_logn = np.zeros_like(m, dtype=float)
        std_logn[mask] = np.sqrt(np.log(1 + v[mask] / np.power(m[mask], 2)))
        # mean_logn = np.log(m / np.sqrt(1 + v / np.power(m, 2)))
        # std_logn = np.sqrt(np.log(1 + v / np.power(m, 2)))
        return mean_logn, std_logn # mean, std of underlying normal distribution

    @staticmethod
    def _extract_vector_distance_duration(df):
        core = df.iloc[:, 1:].copy()
        lower_cols = {str(c).strip().lower(): c for c in core.columns}

        if 'distance' in lower_cols:
            distance = core[lower_cols['distance']].to_numpy(dtype='float32')
        else:
            distance = core.iloc[:, 0].to_numpy(dtype='float32')

        duration = None
        if 'duration' in lower_cols:
            duration = core[lower_cols['duration']].to_numpy(dtype='float32')
        return distance, duration

    @staticmethod
    def _extract_matrix(df):
        return df.iloc[:, 1:].to_numpy(dtype='float32')

    def setup_hospital(self, cfg_hospital):
        reg_prop ={} # hos_num, hos_max_capa, hos_tier, d_HtoH_euc, d_HtoH_road, d_HtoS_euc, d_HtoS_road
        # From data
        if cfg_hospital['load_data']:
            try:
                # Hospital capacity and tier information
                hos_info = pd.read_csv(cfg_hospital['info_path'])
                reg_prop['hos_num'] = len(hos_info)
                # reg_prop['hos_max_capa'] = hos_info['num_beds'].to_numpy(dtype='int32')
                # reg_prop['hos_max_queue'] = hos_info['queue_capa'].to_numpy(dtype='int32')
                # ========== [Modified] num_beds -> num_operating_rooms, queue_capa -> num_beds ==========
                reg_prop['hos_max_capa'] = hos_info['num_or'].to_numpy(dtype='int32')
                reg_prop['hos_max_queue'] = hos_info['num_beds'].to_numpy(dtype='int32')
                reg_prop['hos_tier'] = hos_info['type_code'].to_numpy(dtype='int32')
                reg_prop['hos_max_send'] = cfg_hospital['max_send_coeff'][0]*reg_prop['hos_max_capa'] \
                                           + cfg_hospital['max_send_coeff'][1]*reg_prop['hos_max_queue']
                # Distance information
                d_HtoH_euc = pd.read_csv(cfg_hospital['dist_Hos2Hos_euc_info'])
                reg_prop['d_HtoH_euc'] = self._extract_matrix(d_HtoH_euc)
                d_HtoH_road = pd.read_csv(cfg_hospital['dist_Hos2Hos_road_info'])
                reg_prop['d_HtoH_road'] = self._extract_matrix(d_HtoH_road)

                d_HtoS_euc = pd.read_csv(cfg_hospital['dist_Hos2Site_euc_info']).iloc[:, 1:]
                reg_prop['d_HtoS_euc'] = d_HtoS_euc.to_numpy(dtype='float32')[:,0]
                d_HtoS_road = pd.read_csv(cfg_hospital['dist_Hos2Site_road_info'])
                reg_prop['d_HtoS_road'], reg_prop['t_HtoS_road_api'] = self._extract_vector_distance_duration(d_HtoS_road)
            except FileNotFoundError:
                print("Required files for hospital data generation are missing.")
        else:
            # call generator
            raise NotImplementedError("Scenario generation module import is not yet implemented.")
        # Modify & Add
        reg_prop['hos_tier'] = np.where(reg_prop['hos_tier']==1, 3, 2) # 3 = Advanced general hospital (Tier3), 2 = Others (Tier2)
        reg_prop['hos_tier3_idx'] = hos_info.index[hos_info['type_code'] == 1].to_numpy()
        reg_prop['hos_tier2_idx'] = hos_info.index[hos_info['type_code'] != 1].to_numpy()
        reg_prop['hos_closest_second'] = reg_prop['hos_tier2_idx'][
            np.argmin(reg_prop['d_HtoS_road'][reg_prop['hos_tier2_idx']])] if len(reg_prop['hos_tier2_idx']) else None
        reg_prop['hos_closest_third'] = reg_prop['hos_tier3_idx'][
            np.argmin(reg_prop['d_HtoS_road'][reg_prop['hos_tier3_idx']])] if len(reg_prop['hos_tier3_idx']) else None

        HtoH_road = reg_prop['d_HtoH_road'].copy()
        np.fill_diagonal(HtoH_road, np.inf)
        reg_prop['hos_closest_third_fromH'] = reg_prop['hos_tier3_idx'][np.argmin(HtoH_road[:,reg_prop['hos_tier3_idx']], axis=1)]
        reg_prop['hos_closest_second_fromH'] = reg_prop['hos_tier2_idx'][np.argmin(HtoH_road[:, reg_prop['hos_tier2_idx']], axis=1)]

        # Extract hospital helipad indices
        if 'helipad' in hos_info.columns:
            reg_prop['hos_helipad_idx'] = hos_info.index[hos_info['helipad'] == 1].to_numpy()
        else:
            print("  Warning: hospital_info has no 'helipad' column. Setting helipad index to empty array.")
            reg_prop['hos_helipad_idx'] = np.array([])

        return reg_prop

    def setup_ambulance(self, cfg_amb):
        reg_prop ={} # amb_num, amb_dispatch_d, amb_v, amb_handover_time, amb_e_types
        # From data
        if cfg_amb['load_data']:
            try:
                amb_info = pd.read_csv(cfg_amb['dispatch_distance_info'])
                reg_prop['amb_num'] = len(amb_info)
                reg_prop['amb_dispatch_d'] = amb_info['init_distance'].to_numpy(dtype='float32')

                # Check and load duration column
                if 'duration' in amb_info.columns:
                    reg_prop['amb_dispatch_t'] = amb_info['duration'].to_numpy(dtype='float32')
                else:
                    # Print warning if duration column is missing
                    print("  Warning: amb_info has no duration column. Falling back to distance/speed calculation.")
                    reg_prop['amb_dispatch_t'] = None

                reg_prop['amb_v'] = cfg_amb['velocity']
                reg_prop['amb_handover_time'] = cfg_amb['handover_time']
            except FileNotFoundError:
                print("Required files for ambulance data generation are missing.")
        else:
            # call generator
            raise NotImplementedError("Scenario generation module import is not yet implemented.")

        # Modify & Add
        # 1. Store initial dispatch time parameters
        # Check is_use_time flag (ambulance.is_use_time in YAML)
        use_api_time = cfg_amb.get('is_use_time', False)
        # Check duration_coeff weight (ambulance.duration_coeff in YAML, default: 1.0)
        duration_coeff = cfg_amb.get('duration_coeff', 1.0)

        if use_api_time and reg_prop.get('amb_dispatch_t') is not None:
            # Apply weight to API-provided duration (minutes)
            response_mean = reg_prop['amb_dispatch_t'] * duration_coeff
        else:
            # Distance/speed based calculation (legacy method)
            response_mean = reg_prop['amb_dispatch_d'] * 60 / reg_prop['amb_v']  # unit: minutes

        response_mean_logn, response_std_logn = self.get_lognormal_param(response_mean)
        reg_prop['amb_response_t'] = (response_mean, response_mean_logn, response_std_logn)
        # 2. Store hospital-to-scene travel time parameters
        hospital_prop = self.en_manager.en_properties['hospital']
        if use_api_time and hospital_prop.get('t_HtoS_road_api') is not None:
            transport_HtoS_mean = hospital_prop['t_HtoS_road_api'] * duration_coeff  # unit: minutes
        else:
            if use_api_time and hospital_prop.get('t_HtoS_road_api') is None:
                print("  Warning: distance_Hos2Site_road.csv has no duration column. Use distance/speed for hospital-to-site.")
            transport_HtoS_mean = hospital_prop['d_HtoS_road'] * 60 / reg_prop['amb_v'] # unit: minutes
        transport_HtoS_mean_logn, transport_HtoS_std_logn = self.get_lognormal_param(transport_HtoS_mean)
        reg_prop['amb_HtoS_t'] = (transport_HtoS_mean, transport_HtoS_mean_logn, transport_HtoS_std_logn)
        # 3. Store hospital-to-hospital travel time parameters
        transport_HtoH_mean = hospital_prop['d_HtoH_road'] * 60 / reg_prop['amb_v'] # unit: minutes
        transport_HtoH_mean_logn, transport_HtoH_std_logn = self.get_lognormal_param(transport_HtoH_mean)
        reg_prop['amb_HtoH_t'] = (transport_HtoH_mean, transport_HtoH_mean_logn, transport_HtoH_std_logn)
        # 4. Store max inter-hospital travel distance
        reg_prop['amb_maxD_HtoH'] = np.max(transport_HtoH_mean_logn, None, None)
        return reg_prop

    def setup_uav(self, cfg_uav):
        reg_prop ={} # uav_num, uav_dispatch_d, uav_v, uav_handover_time, uav_e_types
        # From data
        if cfg_uav['load_data']:
            try:
                uav_info = pd.read_csv(cfg_uav['dispatch_distance_info'])
                reg_prop['uav_num'] = len(uav_info)

                # If UAV count is 0, initialize with empty arrays and return early
                if reg_prop['uav_num'] == 0:
                    print("  UAV count is 0. Initializing with default parameters.")
                    reg_prop['uav_dispatch_d'] = np.array([], dtype='float32')
                    reg_prop['uav_v'] = cfg_uav['velocity']
                    reg_prop['uav_handover_time'] = cfg_uav['handover_time']
                    # Initialize with empty parameters
                    reg_prop['uav_response_t'] = (np.array([]), np.array([]), np.array([]))
                    reg_prop['uav_HtoS_t'] = (np.array([]), np.array([]), np.array([]))
                    reg_prop['uav_HtoH_t'] = (np.array([]), np.array([]), np.array([]))
                    reg_prop['uav_maxD_HtoH'] = 0
                    return reg_prop
                                    
                reg_prop['uav_dispatch_d'] = uav_info['init_distance'].to_numpy(dtype='float32')
                reg_prop['uav_v'] = cfg_uav['velocity']
                reg_prop['uav_handover_time'] = cfg_uav['handover_time']
            except FileNotFoundError:
                print("Required files for UAV data generation are missing.")
        else:
            # call generator
            raise NotImplementedError("Scenario generation module import is not yet implemented.")
        # Modify & Add
        # 1. Store initial dispatch time parameters
        response_mean = reg_prop['uav_dispatch_d'] * 60 / reg_prop['uav_v'] # unit: minutes
        response_mean_logn, response_std_logn = self.get_lognormal_param(response_mean)
        reg_prop['uav_response_t'] = (response_mean, response_mean_logn, response_std_logn)
        # 2. Store hospital-to-scene travel time parameters
        transport_HtoS_mean = self.en_manager.en_properties['hospital']['d_HtoS_euc'] * 60 / reg_prop['uav_v'] # unit: minutes
        transport_HtoS_mean_logn, transport_HtoS_std_logn = self.get_lognormal_param(transport_HtoS_mean)
        reg_prop['uav_HtoS_t'] = (transport_HtoS_mean, transport_HtoS_mean_logn, transport_HtoS_std_logn)
        # 3. Store hospital-to-hospital travel time parameters
        transport_HtoH_mean = self.en_manager.en_properties['hospital']['d_HtoH_euc'] * 60 / reg_prop['uav_v'] # unit: minutes
        transport_HtoH_mean_logn, transport_HtoH_std_logn = self.get_lognormal_param(transport_HtoH_mean)
        reg_prop['uav_HtoH_t'] = (transport_HtoH_mean, transport_HtoH_mean_logn, transport_HtoH_std_logn)
        # 4. Store max inter-hospital travel distance
        reg_prop['uav_maxD_HtoH'] = np.max(transport_HtoH_mean_logn, None, None)
        return reg_prop

    ######### Entity addition template ########
    def setup_template(self, cfg):
        reg_prop = {}
        # FILL IN
        return reg_prop

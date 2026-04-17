import heapq
import numpy as np

class EventManager():
    def __init__(self, ev_info, en_manager, rng=None, enable_trace=False):
        self.ev_info = ev_info
        self.en_manager = en_manager
        self.properties = self.en_manager.en_properties
        self.enable_trace = enable_trace
        self.trace_log = []

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.event_queue = []

    def set_seed(self, rng):
        self.rng = rng

    def start(self):
        self.e_ID = 0 # event creation counter initialized
        self.time = 0 # event clock initialized
        self.status = self.en_manager.en_status # load entity status
        self.rescue_finish = False # rescue completion flag
        self.event_queue = [] # event queue initialized
        self.trace_log = []  # reset trace
        # Incident onset
        init_log = {}
        init_log, _ = self.ev_onset(init_log, None)
        # Run until first decision epoch
        auto_log, _ = self.run_next(action=None)
        init_log['p_admit'] = auto_log.get('p_admit', [])
        return init_log

    def run_next(self, action=None):
        log = {'p_admit':[]} # log to record during simulation run
        if action is not None:
            normal_action, repeat = self.proceed_action(action)
            log['normal_action'] = normal_action
            if repeat: # additional action needed
                return log, False

        terminated = False
        while True:
            # 1. Process the earliest event
            if not self.event_queue:

                # If no more events, terminate (same as before)

                h_idle_que_occ = self.status['hospital']['h_states'].copy() if 'hospital' in self.status else None

                if h_idle_que_occ is None:
                    print(f'[EventQueueEmpty] t={self.time} h_states=None')
                else:
                    # Option to prevent numpy from truncating with ...
                    with np.printoptions(threshold=np.inf, linewidth=200, suppress=True):
                        print(f'[EventQueueEmpty] t={self.time}  h_states.shape={h_idle_que_occ.shape}\n{h_idle_que_occ}')

                return log, True

            c_event = heapq.heappop(self.event_queue)  # event = (event_time, e_ID, ev_name, entity_idx)
            print(c_event)

            time_interval = c_event[0] - self.time
            self.time = c_event[0]
            # Update status based on elapsed time
            self.status['ambulance']['amb_states'][:,1] -= time_interval
            np.maximum(self.status['ambulance']['amb_states'][:,1], 0, out=self.status['ambulance']['amb_states'][:,1])
            self.status['uav']['uav_states'][:,1] -= time_interval
            np.maximum(self.status['uav']['uav_states'][:,1], 0, out=self.status['uav']['uav_states'][:,1])
            log, stop_condition = getattr(self, "ev_" + c_event[2])(log, c_event[3])
            if stop_condition: # 2. Run until decision is requested
                break
            terminated = self.check_termination() # 3. If no more decisions needed, run remaining simulation and terminate
            if terminated:
                break
        return log, terminated

    def proceed_action(self, action):
        print("Action:", action)
        # action[0]: Red = 0, Yellow = 1, Green = 2
        # action[1]: 0: site, hospital 1 ~ hospital N; e.g. with 10 hospitals: 0 is site, 1~9 are hospitals
        # action[2]: 0: Amb, 1: UAV
        # return: normal_action, repeat
        p_class, destination, mode = action # destination = hospital index + 1
        if destination == 0:  # stay at site
            return True, False
        else:  # transport to hospital
            # 0. Penalize wrong mode selection & terminate
            if mode == 1 and not self.status['uav']['uav_wait'][0]:  # UAV unavailable, invalid transport command
                print("NO UAV")
                return False, False
            elif mode == 0 and not self.status['ambulance']['amb_wait'][0]:  # Amb unavailable, invalid transport command
                print("NO AMB")
                return False, False

            # 1. Update site patient count
            try:
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
                # # TO-DO: UPDATE when using Rule
                # if p_class == 0 or p_class == 1:
                #     for idx, record in enumerate(self.slack_times):
                #         if record[1] == action[0]:
                #             del self.slack_times[idx]
                #             break
            except IndexError:
                print("NO PATIENT")
                return False, False

            # 2. Update destination and dispatch time based on transport mode
            h_idx = destination - 1
            tranportation_t = self.sample_transportation_time(mode=mode, origination=0, destination=destination)
            if mode == 0: # amb
                a_idx = self.status['ambulance']['amb_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time'] # total patient delivery time
                # Update status
                self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1 # sent record
                self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="AMB",
                                   vehicle_id=int(a_idx), hospital_id=int(h_idx), severity=int(p_class))
            elif mode == 1: # uav
                u_idx = self.status['uav']['uav_wait'][0].pop()
                elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time'] # total patient delivery time
                # Update status
                self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class+1) # destination, time, severity
                self.status['patient']['p_states'][p_idx, 2] = 1  # move
                self.status['patient']['p_sent'][h_idx] += 1  # sent record
                self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, h_idx))
                self._record_trace("transport_start", patient_id=int(p_idx), vehicle="UAV",
                                   vehicle_id=int(u_idx), hospital_id=int(h_idx), severity=int(p_class))

            # 3. If R/Y patients remain at site and transport mode is available, request additional decision
            hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasAvailableMode and hasRY:
                return True, True # request another decision
            else:
                return True, False

    def check_termination(self):
        # If all patients have been treated, terminated (avoids unnecessary iterations)
        terminated = np.all(self.status['patient']['p_states'][:,-1] == 1)
        # or verify via event_queue length
        return terminated

    def sample_transportation_time(self, mode, origination, destination):
        # Note: In data, hospital idx starts from 0; destination and origination hospital idx start from 1
        # origination=0: site, destination=0: site (return trip)
        # 1. Sample transportation time
        log_mean, log_std = None, None
        if origination == 0:  # Site ??Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][destination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][destination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][destination - 1]
        elif destination == 0:  # Hospital -> Site (return trip)
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoS_t'][1][origination - 1]
                log_std = self.properties['ambulance']['amb_HtoS_t'][2][origination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoS_t'][1][origination - 1]
                log_std = self.properties['uav']['uav_HtoS_t'][2][origination - 1]
        else:  # Hospital ??Hospital
            if mode == 0:  # ambulance
                log_mean = self.properties['ambulance']['amb_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['ambulance']['amb_HtoH_t'][2][origination-1, destination - 1]
            elif mode == 1:  # UAV
                log_mean = self.properties['uav']['uav_HtoH_t'][1][origination-1, destination - 1]
                log_std = self.properties['uav']['uav_HtoH_t'][2][origination-1, destination - 1]
        tranportation_t = self.rng.lognormal(log_mean, log_std)
        return tranportation_t

    def start_GB_transport(self, log):
        # Transport Green/Black patients at site using all available transport modes at site
        # 1. Use UAV first
        while self.status['uav']['uav_wait'][0]:
            u_idx = self.status['uav']['uav_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # transport Green waiting at site
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # transport Black waiting at site
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=1)
            tranportation_t = self.sample_transportation_time(mode=1, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['uav']['uav_handover_time']  # total patient delivery time
            # Update status
            self.status['uav']['uav_states'][u_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
        # 2. Use AMB
        while self.status['ambulance']['amb_wait'][0]:
            a_idx = self.status['ambulance']['amb_wait'][0].pop()
            if self.status['patient']['p_wait'][2][0]:  # transport Green waiting at site
                p_class = 2
                p_idx = self.status['patient']['p_wait'][p_class][0].pop()
            elif self.status['patient']['p_wait'][3][0]:  # transport Black waiting at site
                p_class = 3
                p_idx = self.status['patient']['p_wait'][3][0].pop()
            else:
                break
            destination = self.default_transportation_GB(mode=0)
            tranportation_t = self.sample_transportation_time(mode=0, origination=0, destination=destination)
            elapsed_time = tranportation_t + self.properties['ambulance']['amb_handover_time']  # total patient delivery time
            # Update status
            self.status['ambulance']['amb_states'][a_idx] = (destination, elapsed_time, p_class + 1)  # destination, time, severity
            self.status['patient']['p_states'][p_idx, 2] = 1  # move
            self.status['patient']['p_sent'][destination-1] += 1  # sent record
            self.add_event(elapsed_time, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))

        return log

    def default_transportation_GB(self, mode):
        # Rule1: Ver250724
        # 1. Exclude Tier3 (tertiary) hospitals
        # 2. Transport to nearest hospital in order (hospital index sorted by distance from site)
        # 3. Only transport if max_send - p_sent > 0 (max sendable patients - actually sent patients)
        # 4. If no matching hospital, transport to nearest hospital regardless of tier
        # 5. If UAV (mode=1), only transport to hospitals with helipad

        destination = None
        idle_capa = self.properties['hospital']['hos_max_send'] - self.status['patient']['p_sent']
        helipad_idx = self.properties['hospital'].get('hos_helipad_idx', np.array([]))
        for h_idx in self.properties['hospital']['hos_tier2_idx']:
            if mode == 1 and h_idx not in helipad_idx:
                continue
            if idle_capa[h_idx] > 0:
                destination = h_idx + 1
                break
        if destination is None:
            for h_idx in self.properties['hospital']['hos_tier3_idx']:
                if mode == 1 and h_idx not in helipad_idx:
                    continue
                if idle_capa[h_idx] > 0:
                    destination = h_idx + 1
                    break
        if destination is None:
            raise RuntimeError("No hospital with remaining send capacity for Green/Black transport.")
        return destination

    def diversion_rule(self, c_hos, pass_to_tier3, pass_to_tier2, mode):
        # Rule1: Ver250724
        # 1. Transport to nearest hospital in order among treatable-tier hospitals
        # 2. Only transport if max_send - p_sent > 0 (max sendable patients - actually sent patients)
        # 3. If no matching hospital, raise error message
        # 4. If UAV (mode=1), only transport to hospitals with helipad

        d_to_H = self.properties['hospital']['d_HtoH_road'][c_hos] if mode==0 else self.properties['hospital']['d_HtoH_euc'][c_hos]
        destination = None
        idle_capa = self.properties['hospital']['hos_max_send'] - self.status['patient']['p_sent']
        helipad_idx = self.properties['hospital'].get('hos_helipad_idx', np.array([]))
        max_capa_arr = self.properties['hospital']['hos_max_capa'] + self.properties['hospital']['hos_max_queue']
        n_occupied_arr = self.status['hospital']['h_states'][:, -1]

        sorted_h = np.argsort(d_to_H)
        for h_idx in sorted_h:
            if mode == 1 and h_idx not in helipad_idx:
                continue
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            can_admit = (pass_to_tier3 and h_tier==3) or (pass_to_tier2 and h_tier==2)
            if can_admit and idle_capa[h_idx] > 0 and n_occupied_arr[h_idx] < max_capa_arr[h_idx]:
                destination = h_idx + 1
                break
        if destination is None: # no reachable hospital available
            raise Exception("Impossible to divert")

        return destination

    def sample_service_time(self, h_tier, p_class):
        #   if service time 9999 then n_idle -= 1, change to definite cared; no additional event created
        if h_tier == 3:
            service_mean = self.properties['patient']['patient_info']['treat_tier3_mean'][p_class]
        elif h_tier == 2:
            service_mean = self.properties['patient']['patient_info']['treat_tier2_mean'][p_class]
        if isinstance(service_mean, str):
            service_time = np.inf
        else:
            service_time = self.rng.exponential(service_mean)
        return service_time

    def ev_onset(self, log, entity_idx):
        """
        Incident onset event
        :return:
        """
        rescue_times = []
        # 1. Create patient rescue events
        p_param = self.properties['patient']
        p_num = self.rng.multinomial(p_param['incident_size'],
                                     pvals=p_param['patient_info']['ratio'])
        self.status['patient']['p_states'][:,0] = np.repeat([0,1,2,3], p_num)

        rescue_max_time = 60

        for p_class in range(4):
            alpha, beta = p_param['patient_info']['rescue_param_alpha'][p_class], p_param['patient_info']['rescue_param_beta'][p_class]
            if alpha != 0 and beta != 0:
                sampled = self.rng.beta(alpha, beta, size = p_num[p_class]) * rescue_max_time
            else:
                sampled = np.zeros(p_num[p_class])
            rescue_times.append(sampled)
        p_idx = 0
        for p_class, event_times in enumerate(rescue_times):
            for t in event_times:
                self.add_event(elapsed_time=t, ev_name='p_rescue', entity_idx=(p_idx,))
                p_idx += 1
        # 2. Create amb site arrival (dispatch) events
        amb_response_param = self.properties['ambulance']['amb_response_t']
        time_amb = self.rng.lognormal(amb_response_param[1], amb_response_param[2])
        for a_idx, t in enumerate(time_amb):
            self.add_event(elapsed_time=t, ev_name='amb_arrival_site', entity_idx=(a_idx,))
        self.status['ambulance']['amb_states'][:,1] = time_amb
        # 3. Create uav site arrival (dispatch) events
        uav_response_param = self.properties['uav']['uav_response_t']
        time_uav = self.rng.lognormal(uav_response_param[1], uav_response_param[2])
        for u_idx, t in enumerate(time_uav):
            self.add_event(elapsed_time=t, ev_name='uav_arrival_site', entity_idx=(u_idx,))
        self.status['uav']['uav_states'][:,1] = time_uav

        self._record_trace("onset", n_patients=int(sum(p_num)),
                           severity_dist=[int(x) for x in p_num])

        log = {'rescue_times': rescue_times}
        return log, False

    def ev_p_rescue(self, log, entity_idx):
        """
        Patient rescue event
        :param log: event log recording dictionary
        :param entity_idx: tuple(patient idx, )
        :return:
        log
        stop_condition
        """
        p_idx = entity_idx[0]
        p_class = self.status['patient']['p_states'][p_idx, 0]
        self.status['patient']['p_states'][p_idx, 1] = 1 # rescued
        self.status['patient']['p_wait'][p_class][0].append(p_idx)
        self._record_trace("rescue", patient_id=int(p_idx), severity=int(p_class))
        self.rescue_finish = np.all(self.status['patient']['p_states'][:, 1] == 1) # if min value is 0, still un-rescued patients exist

        hasAvailableMode = bool(self.status['ambulance']['amb_wait'][0] or self.status['uav']['uav_wait'][0])
        if not hasAvailableMode: # if no transport mode available, run next event
            return log, False
        else:
            hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
            if hasRY: # transport available and R/Y patients at site -> request decision
                return log, True
            else:
                if self.rescue_finish: # transport available, no R/Y at site, all patients rescued -> start GB transport
                    log = self.start_GB_transport(log)
                return log, False

    def ev_amb_arrival_site(self, log, entity_idx):
        """
        Ambulance site arrival event
        :param log: event log recording dictionary
        :param entity_idx: tuple(ambulance idx, )
        :return:
        """
        a_idx = entity_idx[0]
        self.status['ambulance']['amb_wait'][0].append(a_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. If Red or Yellow patients are at site, request decision
            return log, True
        else: # 2. If no Red or Yellow patients at site
            if self.rescue_finish: # if rescue finished, start Green -> Black patient transport
                log = self.start_GB_transport(log)
                return log, False
            else: # un-rescued patients remain -> wait at site
                return log, False

    def ev_uav_arrival_site(self, log, entity_idx):
        """
        UAV site arrival event
        :param log: event log recording dictionary
        :param entity_idx: tuple(uav idx, )
        :return:
        """
        u_idx = entity_idx[0]
        self.status['uav']['uav_wait'][0].append(u_idx)

        hasRY = bool(self.status['patient']['p_wait'][0][0] or self.status['patient']['p_wait'][1][0])
        if hasRY: # 1. If Red or Yellow patients are at site, request decision
            return log, True
        else: # 2. If no Red or Yellow patients at site
            if self.rescue_finish: # if rescue finished, start Green -> Black patient transport
                log = self.start_GB_transport(log)
                return log, False
            else: # un-rescued patients remain -> wait at site
                return log, False

    def ev_p_care_ready(self, log, entity_idx):
        """
        Hospital arrival and handover / triage / initial treatment completion event
        :param log:
        :param entity_idx:
        :return:
        """
        p_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        if n_idle > 0: # start treatment
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            self._record_trace("care_start", patient_id=int(p_idx), hospital_id=int(h_idx), severity=int(p_class))
            # Update hospital and patient status
            self.status['hospital']['h_states'][h_idx, 0] -= 1  # n_idle -= 1
            # Add event
            if service_time == np.inf: # permanent capacity occupation
                self.status['patient']['p_states'][p_idx, -1] = 1
            else: # add treatment completion event
                self.add_event(service_time, 'p_def_care', (p_idx, h_idx))
        else: # start queuing (events were already added under feasibility conditions so max_queue is not exceeded)
            # Update hospital and patient status
            self.status['hospital']['h_states'][h_idx, 1] += 1  # n_queue += 1
            self.status['patient']['p_wait'][p_class][h_idx+1].append(p_idx) # patient starts queuing
        return log, False

    def _can_treat_patient(self, h_idx, p_class):
        h_tier = self.properties['hospital']['hos_tier'][h_idx]
        p_info = self.properties['patient']['patient_info']
        if h_tier == 3:
            return bool(p_info['treat_tier3'][p_class])
        if h_tier == 2:
            return bool(p_info['treat_tier2'][p_class])
        return False

    def ev_amb_arrival_hospital(self, log, entity_idx):
        """
        Ambulance hospital-arrival event.
        :param log: event log dictionary
        :param entity_idx: tuple(patient idx, ambulance idx, hospital idx)
        :return:
        """
        p_idx, a_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        p_info = self.properties['patient']['patient_info']
        destination = 0
        handover_time = 0

        if not self._can_treat_patient(h_idx, p_class):
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=0)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            transportation_t = self.sample_transportation_time(mode=0, origination=h_idx + 1, destination=destination)
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))
            return log, False

        max_capa = self.properties['hospital']['hos_max_capa'][h_idx] + self.properties['hospital']['hos_max_queue'][h_idx]
        n_occupied = self.status['hospital']['h_states'][h_idx, -1]
        if n_occupied < max_capa:
            self.status['patient']['p_states'][p_idx, 3] = 1
            self.status['hospital']['h_states'][h_idx, -1] += 1
            handover_time = self.properties['ambulance']['amb_handover_time']
            self.add_event(handover_time, 'p_care_ready', (p_idx, h_idx))
            self._record_trace("hospital_arrival", patient_id=int(p_idx), vehicle="AMB",
                               hospital_id=int(h_idx), severity=int(p_class), admitted=True)
        else:
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=0)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            self._record_trace("diversion", patient_id=int(p_idx), vehicle="AMB",
                               from_hospital=int(h_idx), to_hospital=int(destination-1), severity=int(p_class))

        transportation_t = self.sample_transportation_time(mode=0, origination=h_idx + 1, destination=destination)
        if destination == 0:
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, 0)
            self.add_event(transportation_t + handover_time, 'amb_arrival_site', (a_idx,))
        else:
            self.status['ambulance']['amb_states'][a_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t + handover_time, 'amb_arrival_hospital', (p_idx, a_idx, destination - 1))
        return log, False

    def ev_uav_arrival_hospital(self, log, entity_idx):
        """
        UAV hospital-arrival event.
        :param log: event log dictionary
        :param entity_idx: tuple(patient idx, uav idx, hospital idx)
        :return:
        """
        p_idx, u_idx, h_idx = entity_idx
        p_class = self.status['patient']['p_states'][p_idx, 0]
        p_info = self.properties['patient']['patient_info']
        destination = 0
        handover_time = 0

        if not self._can_treat_patient(h_idx, p_class):
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=1)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            transportation_t = self.sample_transportation_time(mode=1, origination=h_idx + 1, destination=destination)
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
            return log, False

        max_capa = self.properties['hospital']['hos_max_capa'][h_idx] + self.properties['hospital']['hos_max_queue'][h_idx]
        n_occupied = self.status['hospital']['h_states'][h_idx, -1]
        if n_occupied < max_capa:
            self.status['patient']['p_states'][p_idx, 3] = 1
            self.status['hospital']['h_states'][h_idx, -1] += 1
            handover_time = self.properties['uav']['uav_handover_time']
            self.add_event(handover_time, 'p_care_ready', (p_idx, h_idx))
            self._record_trace("hospital_arrival", patient_id=int(p_idx), vehicle="UAV",
                               hospital_id=int(h_idx), severity=int(p_class), admitted=True)
        else:
            destination = self.diversion_rule(h_idx,
                                              pass_to_tier3=p_info['treat_tier3'][p_class],
                                              pass_to_tier2=p_info['treat_tier2'][p_class],
                                              mode=1)
            self.status['patient']['p_sent'][h_idx] -= 1
            self.status['patient']['p_sent'][destination-1] += 1
            self._record_trace("diversion", patient_id=int(p_idx), vehicle="UAV",
                               from_hospital=int(h_idx), to_hospital=int(destination-1), severity=int(p_class))

        transportation_t = self.sample_transportation_time(mode=1, origination=h_idx + 1, destination=destination)
        if destination == 0:
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, 0)
            self.add_event(transportation_t + handover_time, 'uav_arrival_site', (u_idx,))
        else:
            self.status['uav']['uav_states'][u_idx] = (destination, transportation_t, p_class + 1)
            self.add_event(transportation_t + handover_time, 'uav_arrival_hospital', (p_idx, u_idx, destination - 1))
        return log, False

    def ev_p_def_care(self, log, entity_idx):
        """
        Hospital patient treatment completion event
        :param log: event log recording dictionary
        :param entity_idx: tuple(patient idx, hospital idx)
        :return:
        """
        p_idx, h_idx = entity_idx
        # Update treated patient status
        self.status['patient']['p_states'][p_idx, -1] = 1
        self._record_trace("care_complete", patient_id=int(p_idx), hospital_id=int(h_idx))

        n_idle, n_queue = self.status['hospital']['h_states'][h_idx][0:2]
        # Start new treatment
        if n_queue > 0:
            h_tier = self.properties['hospital']['hos_tier'][h_idx]
            # Treat in order: Red, Yellow, Green, Black
            for p_class in range(4):
                if self.status['patient']['p_wait'][p_class][h_idx+1]:
                    new_p_idx = self.status['patient']['p_wait'][p_class][h_idx+1].pop()
                    break
            service_time = self.sample_service_time(h_tier=h_tier, p_class=p_class)
            log['p_admit'].append((self.time, p_class))
            # Update hospital and patient status
            self.status['hospital']['h_states'][h_idx, 1] -= 1  # n_queue -= 1
            # Add event
            if service_time == np.inf: # permanent capacity occupation
                self.status['patient']['p_states'][new_p_idx, -1] = 1
            else: # add treatment completion event
                self.add_event(service_time, 'p_def_care', (new_p_idx, h_idx))
        else:
            self.status['hospital']['h_states'][h_idx, 0] += 1  # n_idle += 1
        return log, False

    def _record_trace(self, event_type, **kwargs):
        """Record a trace event if tracing is enabled."""
        if self.enable_trace:
            record = {"time": self.time, "event": event_type}
            record.update(kwargs)
            self.trace_log.append(record)

    def get_trace(self):
        """Return the accumulated trace log."""
        return list(self.trace_log)

    def add_event(self, elapsed_time, ev_name, entity_idx):
        self.e_ID += 1
        heapq.heappush(self.event_queue, (elapsed_time + self.time, self.e_ID, ev_name, entity_idx)) # event = (event_time, e_ID, ev_name, entity_idx)

    def ev_template(self, log, entity_idx):
        """
        :param log: event log recording dictionary
        :param entity_idx: index for each entity participating in the event
        :return:
        stop_condition: flag for pausing to request decision
        """
        stop_condition = False
        return log, stop_condition



import numpy as np
class RuleManager():
    def __init__(self, configs, scenario, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self.scenario = scenario
        self.rules = []
        self.rule_names = []
        if configs['isFullFactorial']:
            for priority in ["START", "ReSTART"]:
                # for hos_select in ["RedOnly", "YellowHalf"]:
                for hos_select in ["RedOnly", "YellowNearest"]:
                    for mode_R in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]:
                        for mode_Y in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]:
                            self.rules.append(Universal_Rule(priority, hos_select, mode_R, mode_Y))
                            self.rules[-1].set_seed(self.rng)
                            self.rules[-1].init_with_scenario(self.scenario)
        else:
            for priority in configs['priority_rule']:
                for hos_select in configs['hos_select_rule']:
                    for mode_R in configs['red_mode_rule']:
                        for mode_Y in configs['yellow_mode_rule']:
                            self.rules.append(Universal_Rule(priority, hos_select, mode_R, mode_Y))
                            self.rules[-1].set_seed(self.rng)
                            self.rules[-1].init_with_scenario(self.scenario)

    def set_seed(self, rng):
        self.rng = rng

class Rule:
    def __init__(self):
        self.name = "Undefined Rule"
    def init_with_scenario(self, scenario):
        en_properties = scenario['EntityManager'].en_properties

        
    #     # Calculate theta from the mean round-trip travel time to the 3 nearest hospitals
    #     self.theta_amb = np.mean(en_properties['ambulance']['amb_HtoS_t'][0][0:3]) * 2
    #     self.theta_uav = np.mean(en_properties['uav']['uav_HtoS_t'][0][0:3]) * 2
    #     self.K_amb = en_properties['ambulance']['amb_num']
    #     self.K_uav = en_properties['uav']['uav_num']
    
    # Check UAV count first
        self.K_uav = en_properties['uav']['uav_num']
        self.K_amb = en_properties['ambulance']['amb_num']
    
    # AMB theta calculation (always computed; if >=3 vehicles use top-3 mean as above, otherwise use overall mean; 0 if empty)
        amb_distances = en_properties['ambulance']['amb_HtoS_t'][0]
        if len(amb_distances) >= 3:
            self.theta_amb = np.mean(amb_distances[0:3]) * 2
        else:
            self.theta_amb = np.mean(amb_distances) * 2 if len(amb_distances) > 0 else 0
    
    # UAV theta calculation (handles UAV=0 case)
        if self.K_uav == 0:
            self.theta_uav = 0  # Set to 0 when no UAVs available
        else:
            uav_distances = en_properties['uav']['uav_HtoS_t'][0]
            if len(uav_distances) >= 3:
                self.theta_uav = np.mean(uav_distances[0:3]) * 2
            else:
                self.theta_uav = np.mean(uav_distances) * 2 if len(uav_distances) > 0 else 0

        self.expected_R = en_properties['patient']['incident_size'] * en_properties['patient']['patient_info']['ratio'][0]
        self.expected_Y = en_properties['patient']['incident_size'] * en_properties['patient']['patient_info']['ratio'][1]

        self.hos_num = en_properties['hospital']['hos_num']
        self.tier3_idx = en_properties['hospital']['hos_tier3_idx'] # Tier3 = tertiary hospital, Tier2 = others
        self.tier2_idx = en_properties['hospital']['hos_tier2_idx'] # Tier3 = tertiary hospital, Tier2 = others
        self.helipad_idx = en_properties['hospital'].get('hos_helipad_idx', np.array([]))

        self.hos_max_send = en_properties['hospital']['hos_max_send'] # Max number of patients to send (target)


    def set_seed(self, rng):
        self.rng = rng

    def select(self, obs):
        """
        :param obs: Current state information
        :return: Selected decision (action)
        """
        raise NotImplementedError

# Base rule

class Universal_Rule(Rule):
    def __init__(self, priority, hos_select, mode_R, mode_Y):
        assert priority in ["START", "ReSTART"]
        # assert hos_select in ["RedOnly", "YellowHalf"]
        assert hos_select in ["RedOnly", "YellowNearest"]
        assert mode_R in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
        assert mode_Y in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]

        self.priority = priority
        self.hos_select = hos_select
        self.mode_R = mode_R
        self.mode_Y = mode_Y

        self.rule_name = f"{priority}, {hos_select}, Red {mode_R}, Yellow {mode_Y}"

    def select(self, obs):
        self.obs = obs
        action = [0, 0, 0] # STAY action if not changed at all

        # For ReSTART
        if self.priority == "ReSTART":
            mask_red = self.obs['p_states'][:,0] == 0
            mask_yellow = self.obs['p_states'][:,0] == 1
            # red_move = self.obs['p_states'][mask_red][:, 1:].sum()
            # yellow_move = self.obs['p_states'][mask_yellow][:, 1:].sum()

            # p_states[:,0]=class, [:,2]=> move (changed to transport start)
            red_move = self.obs['p_states'][mask_red][:, 2].sum()
            yellow_move = self.obs['p_states'][mask_yellow][:, 2].sum()


            num_D = max(self.expected_Y - yellow_move,0) # Expected remaining yellow patients; yellow_count = yellow patients transported
            num_I = max(self.expected_R - red_move,0)
            # self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
            if self.K_uav > 0:
                self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
            else:
                self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb))
            

        red_exist = self.obs['p_wait'][0][0]
        yellow_exist = self.obs['p_wait'][1][0]
        if not red_exist and not yellow_exist:
            return action
        # num_D root-cause debugging (ReSTART only)
        if self.priority == "ReSTART":
            import sys
            import numpy as np

            
            expected_Y = float(self.expected_Y)
            expected_R = float(self.expected_R)

            red_move_sum = float(red_move)
            yellow_move_sum = float(yellow_move)

            # Alternative metric counting move by "headcount" (sum may overcount if a single patient has 1 in multiple state columns)
            red_move_any = int((self.obs['p_states'][mask_red][:, 1:] > 0).any(axis=1).sum())
            yellow_move_any = int((self.obs['p_states'][mask_yellow][:, 1:] > 0).any(axis=1).sum())

            # Actual number of patients currently waiting in p_wait (queue)
            def _safe_len(x):
                try:
                    return len(x)
                except Exception:
                    return int(bool(x))

            red_wait_n = _safe_len(red_exist)
            yellow_wait_n = _safe_len(yellow_exist)

            # num_D/num_I 
            num_D_calc = max(expected_Y - yellow_move_sum, 0.0)
            num_I_calc = max(expected_R - red_move_sum, 0.0)

            # Anomaly flags
            flag_wait_mismatch = (yellow_wait_n > 0 and num_D_calc == 0.0)                # Patients in queue but calculated as 0
            flag_move_over = (expected_Y > 0 and yellow_move_sum >= expected_Y)           # move exceeds expected
            flag_sum_exploding = (yellow_move_sum > yellow_move_any + 1e-6)               # sum exceeds headcount (any)

            # Prevent duplicate output (avoid log explosion)
            t = float(self.obs['time'])
            key = (int(t), yellow_wait_n, int(flag_wait_mismatch), int(flag_move_over), int(flag_sum_exploding))
            if not hasattr(self, "_nd_dbg_prev"):
                self._nd_dbg_prev = None

            if (flag_wait_mismatch or flag_move_over or flag_sum_exploding) and key != self._nd_dbg_prev:
                y_col_sums = self.obs['p_states'][mask_yellow][:, 1:].sum(axis=0)
                r_col_sums = self.obs['p_states'][mask_red][:, 1:].sum(axis=0)

                print(
                    f"[ND-TRACE] t={t:.2f} | wait(R,Y)=({red_wait_n},{yellow_wait_n}) | "
                    f"expected(R,Y)=({expected_R:.2f},{expected_Y:.2f}) | "
                    f"move_sum(R,Y)=({red_move_sum:.2f},{yellow_move_sum:.2f}) | "
                    f"move_any(R,Y)=({red_move_any},{yellow_move_any}) | "
                    f"num_I={num_I_calc:.2f} num_D={num_D_calc:.2f} | "
                    f"flags(wait_mismatch={flag_wait_mismatch}, move_over={flag_move_over}, sum_exploding={flag_sum_exploding})",
                    file=sys.stderr
                )
                print(f"[ND-TRACE] yellow_col_sums={np.array2string(y_col_sums, precision=0)}", file=sys.stderr)
                print(f"[ND-TRACE] red_col_sums={np.array2string(r_col_sums, precision=0)}", file=sys.stderr)

                self._nd_dbg_prev = key
    


        # 1. Prioirty selection
        if self.priority == "START":
            if red_exist:  # If Red patients exist
                action[0] = 0  # Red
            elif yellow_exist:  # If Yellow patients exist
                action[0] = 1
        elif self.priority == "ReSTART":
            if self.tau <= 0:  # All yellow first, then red
                if yellow_exist:  # If Yellow patients exist
                    action[0] = 1
                elif red_exist:  # If Red patients exist
                    action[0] = 0  # Red
            # elif self.tau >= num_I * (self.theta_amb / self.K_amb + self.theta_uav / self.K_uav):  # Send all red first, then yellow
            # Fix: handle UAV=0 case
            elif (self.K_uav > 0 and 
                  self.tau >= num_I * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav)) or \
                 (self.K_uav == 0 and 
                  self.tau >= num_I * (self.theta_amb/self.K_amb)):
                if red_exist:  # If Red patients exist
                    action[0] = 0  # Red
                elif yellow_exist:  # If Yellow patients exist
                    action[0] = 1
            else:  # Send red patients first until time reaches tau or no red remain on scene, then send all yellow, then remaining
                if self.obs['time'] <= self.tau:
                    if red_exist:  # If Red patients exist
                        action[0] = 0  # Red
                    elif yellow_exist:  # If Yellow patients exist
                        action[0] = 1
                else:
                    if yellow_exist:  # If Yellow patients exist
                        action[0] = 1
                    elif red_exist:  # If Red patients exist
                        action[0] = 0  # Red
        


            

        # 3. Mode selection
        available_UAV = self.obs['uav_wait'][0]
        available_Amb = self.obs['amb_wait'][0]
        if not available_UAV and not available_Amb:
            return action
        isSTAY = False
        if action[0] == 0: # Red selected
            if self.mode_R == "OnlyUAV":
                # if available_UAV: action[2] = 1  # UAV
                # else: action[1] = 0 # STAY
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb:
                    if yellow_exist and self.mode_Y != "OnlyUAV": # Send yellow via AMB
                        action[0] = 1 # Yellow
                        action[2] = 0 # AMB
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "Both_UAVFirst":
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb: action[2] = 0  # AMB
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "Both_AMBFirst":
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV: action[2] = 1  # UAV
                else: print("Error in Transition", action, self.obs)
            elif self.mode_R == "OnlyAMB":
                # if available_Amb: action[2] = 0  # AMB
                # else: action[1] = 0  # STAY
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV:
                    if yellow_exist and self.mode_Y != "OnlyAMB": # Send yellow via UAV
                        action[0] = 1 # Yellow
                        action[2] = 1 # UAV
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
        elif action[0] == 1: # Yellow selected
            if self.mode_Y == "OnlyUAV":
                # if available_UAV: action[2] = 1  # UAV
                # else: action[1] = 0 # STAY

                if available_UAV: action[2] = 1  # UAV
                elif available_Amb:
                    if yellow_exist and self.mode_R != "OnlyUAV": # Send red via AMB
                        action[0] = 0 # Red
                        action[2] = 0 # AMB
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)

            elif self.mode_Y == "Both_UAVFirst":
                if available_UAV: action[2] = 1  # UAV
                elif available_Amb: action[2] = 0  # AMB
                else: print("Error in Transition", action, self.obs)
            elif self.mode_Y == "Both_AMBFirst":
                if available_Amb: action[2] = 0  # AMB
                elif available_UAV: action[2] = 1  # UAV
                else: print("Error in Transition", action, self.obs)
            elif self.mode_Y == "OnlyAMB":
                # if available_Amb: action[2] = 0  # AMB
                # else: action[1] = 0  # STAY

                if available_Amb: action[2] = 0  # AMB
                elif available_UAV:
                    if red_exist and self.mode_R != "OnlyAMB": # Send red via UAV
                        action[0] = 0 # Red
                        action[2] = 1 # UAV
                    else:
                        isSTAY = True # STAY
                else: print("Error in Transition", action, self.obs)
        # 2. Hospital selection
        if not isSTAY:
            if self.hos_select == "RedOnly":
                if action[0] == 0: # Red selected
                    for i in self.tier3_idx:
                        # Helipad check added (when UAV is selected)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
                            action[1] = i + 1
                            break
                elif action[0] == 1: # Yellow selected
                    for i in range(self.hos_num):
                        if i in self.tier3_idx:
                            continue
                        # Helipad check added (when UAV is selected)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
                            action[1] = i + 1
                            break
            # elif self.hos_select == "YellowHalf":
            #     if action[0] == 0:  # Red selected
            #         for i in self.tier3_idx:
            #             # Helipad check added (when UAV is selected)
            #             if action[2] == 1 and i not in self.helipad_idx:
            #                 continue
            #             if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
            #                 action[1] = i + 1
            #                 break
            #     elif action[0] == 1: # Yellow selected
            #         # Get random number from environment for seed control
            #         r = self.rng.random()
            #         if r > 0.5: # Send to tier2
            #             for i in range(self.hos_num):
            #                 if i in self.tier3_idx:  # Tier3 skip - use Tier2 only
            #                     continue
            #                 # Helipad check added (when UAV is selected)
            #                 if action[2] == 1 and i not in self.helipad_idx:
            #                     continue
            #                 if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
            #                     action[1] = i + 1
            #                     break
            #         else: # Send to tier3
            #             for i in self.tier3_idx:  # Use Tier3 only
            #                 # Helipad check added (when UAV is selected)
            #                 if action[2] == 1 and i not in self.helipad_idx:
            #                     continue
            #                 if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
            #                     action[1] = i + 1
            #                     break
            elif self.hos_select == "YellowNearest":
                if action[0] == 0:  # Red selected
                    for i in self.tier3_idx:
                        # Helipad check added (when UAV is selected)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
                            action[1] = i + 1
                            break
                elif action[0] == 1: # Yellow selected - select by distance regardless of tier
                    for i in range(self.hos_num):
                        # Helipad check added (when UAV is selected)
                        if action[2] == 1 and i not in self.helipad_idx:
                            continue
                        if self.hos_max_send[i] > self.obs['h_states'][i,-1]: # max_send > n_occupied
                            action[1] = i + 1
                            break
            

        if isSTAY: # STAY
            action[0], action[2] = -1, -1 # To make redundant

        return action

# class Proposed(Rule):
#     def __init__(self, priority, hos_select, mode_R, mode_Y):
#         assert priority in ["START", "ReSTART"]
#         assert hos_select in ["RedOnly", "YellowHalf"]
#         assert mode_R in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#         assert mode_Y in ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#
#         self.priority = priority
#         self.hos_select = hos_select
#         self.mode_R = mode_R
#         self.mode_Y = mode_Y
#
#         self.rule_name = f"Proposed"
#
#
#     def select(self, obs):
#         self.obs = obs
#         action = [0, 0, 0] # STAY action if not changed at all
#         amb_idx = np.where(np.isclose(self.env.state['time_amb'], 0))
#         uav_idx = np.where(np.isclose(self.env.state['time_uav'], 0))
#         # For ReSTART
#         num_D = self.env.yellow_check - self.env.yellow_count
#         num_I = self.env.red_check - self.env.red_count
#         if num_D <= 0:
#             num_D = 0
#         if num_I <= 0:
#             num_I = 0
#         self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
#
#         # 1. Prioirty selection
#         if self.priority == "START":
#             if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                 action[0] = 0  # Red
#             elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                 action[0] = 1
#         elif self.priority == "ReSTART":
#             if self.tau <= 0:  # All yellow first, then red
#                 if self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                     action[0] = 1
#                 elif self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                     action[0] = 0  # Red
#             elif self.tau >= num_I * (self.theta_amb / self.K_amb + self.theta_uav / self.K_uav):  # Send all red first, then yellow
#                 if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                     action[0] = 0  # Red
#                 elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                     action[0] = 1
#             else:  # Send red patients first until time reaches tau or no red remain on scene, then send all yellow, then remaining
#                 if self.env.time <= self.tau:
#                     if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                         action[0] = 0  # Red
#                     elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                         action[0] = 1
#                 else:
#                     if self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                         action[0] = 1
#                     elif self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                         action[0] = 0  # Red
#
#         # 3. Mode selection
#         isUAV = bool(len(uav_idx[0]))
#         isAmb = bool(len(amb_idx[0]))
#         isSTAY = False
#         if action[0] == 0: # Red selected
#             if self.mode_R == "OnlyUAV":
#                 # if isUAV: action[2] = 1  # UAV
#                 # else: action[1] = 0 # STAY
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb:
#                     if self.obs['p_at_sites'][1] != 0 and self.mode_Y != "OnlyUAV": # Send yellow via AMB
#                         action[0] = 1 # Yellow
#                         action[2] = 0 # AMB
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "OnlyAMB":
#                 # if isAmb: action[2] = 0  # AMB
#                 # else: action[1] = 0  # STAY
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV:
#                     if self.obs['p_at_sites'][1] != 0 and self.mode_Y != "OnlyAMB": # Send yellow via UAV
#                         action[0] = 1 # Yellow
#                         action[2] = 1 # UAV
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#         elif action[0] == 1: # Yellow selected
#             if self.mode_Y == "OnlyUAV":
#                 # if isUAV: action[2] = 1  # UAV
#                 # else: action[1] = 0 # STAY
#
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb:
#                     if self.obs['p_at_sites'][0] != 0 and self.mode_R != "OnlyUAV": # Send red via AMB
#                         action[0] = 0 # Red
#                         action[2] = 0 # AMB
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#
#             elif self.mode_Y == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "OnlyAMB":
#                 # if isAmb: action[2] = 0  # AMB
#                 # else: action[1] = 0  # STAY
#
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV:
#                     if self.obs['p_at_sites'][0] != 0 and self.mode_R != "OnlyAMB": # Send red via UAV
#                         action[0] = 0 # Red
#                         action[2] = 1 # UAV
#                     else:
#                         isSTAY = True # STAY
#                 else: print("Error in Transition", action, self.env.state)
#         # 2. Hospital selection
#         if not isSTAY:
#             if self.hos_select == "RedOnly":
#                 if action[0] == 0: # Red selected
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 elif action[0] == 1: # Yellow selected
#                     for i in range(self.env.numH):
#                         if i in self.env.tier2_idx:
#                             continue
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#             elif self.hos_select == "YellowHalf":
#                 if action[0] == 0:  # Red selected
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 elif action[0] == 1: # Yellow selected
#                     # Get random number from environment for seed control
#                     r = self.env.random_gen.rand()
#                     if r > 0.5: # Send to tier2
#                         for i in self.env.tier2_idx:
#                             if self.env.capa_scale[i] > 0:
#                                 action[1] = i + 1
#                                 break
#                     else: # Send to tier3
#                         for i in range(self.env.numH):
#                             if i in self.env.tier2_idx:
#                                 continue
#                             if self.env.capa_scale[i] > 0:
#                                 action[1] = i + 1
#                                 break
#
#         row_amb = len(self.env.ed_data_amb) - 1
#         row_uav = len(self.env.ed_data_uav) - 1
#         if action[2] == 0: # When AMB is selected
#             # # Version 1: Start when fewer patients than UAV count
#             # if self.env.rescue_finish: # Rescue finished
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num: # Remaining patients fewer than UAV count
#             #         if len(self.obs['time_uav'][self.obs['dest_uav'] == 0]) != 0:
#             #             min_time_uav = min(self.obs['time_uav'][self.obs['dest_uav'] == 0])
#             #         else:
#             #             min_time_uav = np.infty
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # # Version 2: Start when fewer patients than AMB + UAV count
#             # if self.env.rescue_finish: # Rescue finished
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num + self.env.amb_num: # Remaining patients fewer than UAV + AMB count
#             #         if len(self.obs['time_uav'][self.obs['dest_uav'] == 0]) != 0:
#             #             min_time_uav = min(self.obs['time_uav'][self.obs['dest_uav'] == 0])
#             #         else:
#             #             min_time_uav = np.infty
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # # Version 3: Calculate return time
#             # if self.env.rescue_finish: # Rescue finished
#             #     remainedRY = self.env.totalN - self.env.moved_N - self.obs['p_at_sites'][2]
#             #     if remainedRY < self.env.uav_num: # Remaining patients fewer than UAV count
#             #         min_time_uav = np.infty
#             #         for i in range(self.env.uav_num):
#             #             tmp = 0
#             #             if self.obs['dest_uav'][i] == 0:
#             #                 tmp = self.obs['time_uav'][i]
#             #             else:
#             #                 tmp = self.obs['time_uav'][i] + self.env.uav_d_mean[0, self.obs['dest_uav'][i] - 1] * 60 / self.env.v_uav
#             #             if tmp < min_time_uav:
#             #                 min_time_uav = tmp
#             #         time_amb = float(self.env.ed_data_amb.iloc[row_amb, action[1]]) * 60 / self.env.v_amb
#             #         time_uav = float(self.env.ed_data_uav.iloc[row_uav, action[1]]) * 60 / self.env.v_uav
#             #         if min_time_uav < time_amb - time_uav:
#             #             action[1] = 0  # STAY (AMB)
#             # Version 4: Using proposition
#             # if self.env.rescue_finish:  # Rescue finished
#             uav_return_times = sorted(self.env.uav_expected_return) # Expected return-to-scene time for each UAV
#             stay_amb = True
#             for idx, record in enumerate(self.env.slack_times):
#                 if record[0] < uav_return_times[0]: # slack vs expected UAV return time
#                     stay_amb = False
#                     break
#                 else:
#                     uav_return_times[0] += 2 * self.env.uav_d_mean[0, record[2]]
#                 uav_return_times.sort()
#             if stay_amb:
#                 isSTAY = True # STAY (AMB)
#             else:  # Situation where AMB must be used
#                 # Check whether flip priority
#                 uav_return_times = sorted(self.env.uav_expected_return)
#                 isFlip = True
#                 for idx, record in enumerate(self.env.slack_times):
#                     if record[1] == action[0]:
#                         if record[0] < uav_return_times[0]:  # slack vs expected UAV return time
#                             isFlip = False
#                             break
#                         else:
#                             uav_return_times[0] += 2 * self.env.uav_d_mean[0, record[2]]
#                         uav_return_times.sort()
#                 if isFlip: # Priority change
#                     if self.obs['p_at_sites'][abs(action[0] - 1)] != 0:
#                         action[0] = abs(action[0] - 1)
#                         action[2] = 0  # AMB
#                         for idx, record in enumerate(self.env.slack_times):
#                             if record[1] == action[0]:
#                                 action[1] = record[2] + 1
#                                 break
#                 # Version 1
#                 # min_uav_return = min(self.env.uav_expected_return)
#                 # isFlip = False
#                 # for idx, record in enumerate(self.env.slack_times):
#                 #     if record[1] == action[0]:
#                 #         if record[0] > min_uav_return:
#                 #             isFlip = True
#                 #         break
#                 # if isFlip: # Priority change
#                 #     if self.obs['p_at_sites'][abs(action[0] - 1)] != 0:
#                 #         action[0] = abs(action[0] - 1)
#                 #         action[2] = 0  # AMB
#                 #         for idx, record in enumerate(self.env.slack_times):
#                 #             if record[1] == action[0]:
#                 #                 action[1] = record[2] + 1
#                 #                 break
#         if isSTAY: # STAY
#             action[0], action[2] = -1, -1 # To make redundant
#             action[1] = 0
#
#         return action


# class Universial_Rule_for_RL(Rule):
#     def __init__(self):
#         self.priority_list = ["START", "ReSTART"]
#         self.hos_select_list = ["RedOnly", "YellowHalf"]
#         self.mode_R_list = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#         self.mode_Y_list = ["OnlyUAV", "Both_UAVFirst", "Both_AMBFirst", "OnlyAMB"]
#
#     def select_with_args(self, obs, priority, hos_select, mode_R, mode_Y):
#         self.priority = self.priority_list[priority]
#         self.hos_select = self.hos_select_list[hos_select]
#         self.mode_R = self.mode_R_list[mode_R]
#         self.mode_Y = self.mode_Y_list[mode_Y]
#
#         self.obs = obs
#         action = [0, 0, 0] # STAY action if not changed at all
#         amb_idx = np.where(np.isclose(self.env.state['time_amb'], 0))
#         uav_idx = np.where(np.isclose(self.env.state['time_uav'], 0))
#         # For ReSTART
#         num_D = self.env.yellow_check - self.env.yellow_count
#         num_I = self.env.red_check - self.env.red_count
#         if num_D <= 0:
#             num_D = 0
#         if num_I <= 0:
#             num_I = 0
#         self.tau = 71 - (0.5 * num_D * (self.theta_amb/self.K_amb + self.theta_uav/self.K_uav))
#
#         # 1. Prioirty selection
#         if self.priority == "START":
#             if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                 action[0] = 0  # Red
#             elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                 action[0] = 1
#         elif self.priority == "ReSTART":
#             if self.tau <= 0:  # All yellow first, then red
#                 if self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                     action[0] = 1
#                 elif self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                     action[0] = 0  # Red
#             elif self.tau >= num_I * (self.theta_amb / self.K_amb + self.theta_uav / self.K_uav):  # Send all red first, then yellow
#                 if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                     action[0] = 0  # Red
#                 elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                     action[0] = 1
#             else:  # Send red patients first until time reaches tau or no red remain on scene, then send all yellow, then remaining
#                 if self.env.time <= self.tau:
#                     if self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                         action[0] = 0  # Red
#                     elif self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                         action[0] = 1
#                 else:
#                     if self.obs['p_at_sites'][1] != 0:  # If Yellow patients exist
#                         action[0] = 1
#                     elif self.obs['p_at_sites'][0] != 0:  # If Red patients exist
#                         action[0] = 0  # Red
#
#         # 2. Hospital selection
#         if self.hos_select == "RedOnly":
#             if action[0] == 0: # Red selected
#                 for i in self.env.tier2_idx:
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#             elif action[0] == 1: # Yellow selected
#                 for i in range(self.env.numH):
#                     if i in self.env.tier2_idx:
#                         continue
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#         elif self.hos_select == "YellowHalf":
#             if action[0] == 0:  # Red selected
#                 for i in self.env.tier2_idx:
#                     if self.env.capa_scale[i] > 0:
#                         action[1] = i + 1
#                         break
#             elif action[0] == 1: # Yellow selected
#                 # Get random number from environment for seed control
#                 r = self.env.random_gen.rand()
#                 if r > 0.5: # Send to tier2
#                     for i in self.env.tier2_idx:
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#                 else: # Send to tier3
#                     for i in range(self.env.numH):
#                         if i in self.env.tier2_idx:
#                             continue
#                         if self.env.capa_scale[i] > 0:
#                             action[1] = i + 1
#                             break
#
#         # 3. Mode selection
#         isUAV = bool(len(uav_idx[0]))
#         isAmb = bool(len(amb_idx[0]))
#         if action[0] == 0: # Red selected
#             if self.mode_R == "OnlyUAV":
#                 if isUAV: action[2] = 1  # UAV
#                 else: action[1] = 0 # STAY
#             elif self.mode_R == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_R == "OnlyAMB":
#                 if isAmb: action[2] = 0  # AMB
#                 else: action[1] = 0  # STAY
#         elif action[0] == 1: # Yellow selected
#             if self.mode_Y == "OnlyUAV":
#                 if isUAV: action[2] = 1  # UAV
#                 else: action[1] = 0 # STAY
#             elif self.mode_Y == "Both_UAVFirst":
#                 if isUAV: action[2] = 1  # UAV
#                 elif isAmb: action[2] = 0  # AMB
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "Both_AMBFirst":
#                 if isAmb: action[2] = 0  # AMB
#                 elif isUAV: action[2] = 1  # UAV
#                 else: print("Error in Transition", action, self.env.state)
#             elif self.mode_Y == "OnlyAMB":
#                 if isAmb: action[2] = 0  # AMB
#                 else: action[1] = 0  # STAY
#
#         if action[1] == 0: # STAY
#             action[0], action[2] = -1, -1 # To make redundant
#         return action

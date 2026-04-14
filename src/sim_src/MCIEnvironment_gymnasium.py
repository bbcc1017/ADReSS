import gymnasium as gym
import numpy as np
import math

# from gym import spaces
# import pandas as pd

class MCIEnvironment_gym(gym.Env):
    def __init__(self, scenario, rng=None, default_rule = None, max_steps=2000, rule_test=False, eval_mode=False, **kwargs):
        # max_steps: maximum number of decisions (prevents infinite loops without episode termination)
        self.scenario_decoder(scenario)
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.default_rule = default_rule
        self.max_steps = max_steps
        self.rule_test = rule_test
        self.eval_mode = eval_mode
        self.pen_size = 1.0  # Default: 1.0
        self.reset()

    def set_seed(self, rng):
        self.rng = rng

    def scenario_decoder(self, scenario):
        # Entity management
        self.en_manager = scenario['EntityManager']

        # Event management
        self.ev_manager = scenario['EventManager']

    def step(self, action):
        info = {}
        if self.ev_manager.check_termination():
            reward = self.pending_terminal_reward
            self.pending_terminal_reward = 0.0
            if self.rule_test:
                obs = self.en_manager.get_full_obs()
            else:
                obs = self.en_manager.get_obs()
            obs['time'] = self.ev_manager.time
            info['time'] = self.ev_manager.time
            return obs, reward, True, False, info
        self.n_step += 1
        if self.n_step > self.max_steps: # Force termination when max decision count is exceeded
            print("OVERTIME")
            return self.en_manager.get_obs(), -self.pen_size, True, True, info
        log, terminated = self.ev_manager.run_next(action)
        reward = self.logToReward(log)
        if self.rule_test:
            obs = self.en_manager.get_full_obs()
        else:
            obs = self.en_manager.get_obs()
        # Add elements beyond resource state
        obs['time'] = self.ev_manager.time
        info['time'] = self.ev_manager.time
        return obs, reward, terminated, False, info

    def reset(self, seed = None):
        self.pending_terminal_reward = 0.0
        info = {}
        self.n_step = 0 # Decision count

        # Initialize resources
        self.en_manager.init_en_status()
        # Initialize EventManager
        init_log = self.ev_manager.start()
        self.preventable = self.computePreventable(init_log)
        if self.ev_manager.check_termination():
            self.pending_terminal_reward = self.logToReward(init_log)
        # Generate observation
        if self.rule_test:
            obs = self.en_manager.get_full_obs()
        else:
            obs = self.en_manager.get_obs()
        obs['time'] = self.ev_manager.time
        return obs, info

    def computePreventable(self, log):
        val = 0
        for p_class, times in enumerate(log['rescue_times']):
            for t in times:
                val += self.getSurvProb(t, p_class)
        return val

    def logToReward(self, log):
        reward = 0
        for x in log['p_admit']:
            reward += self.getReward(x[0], x[1])
        return reward

    def getReward(self, time, p_class):
        if self.eval_mode:
            val = self.getSurvProb(time, p_class)
        else:
            # 1. SurvProb
            val = self.getSurvProb(time, p_class)
            # # 2. PDR w.o. Green
            # if p_class != 2:
            #     val = self.getSurvProb(time, p_class)/ self.preventable
            # else: # Exclude Green during training
            #     val = 0
        return val

    def getSurvProb(self, time, p_class):
        # read survival probability function
        if (p_class == 0):  # immediate (Red)
            val = 0.56 / (math.pow((time / 91), 1.58) + 1)  # Scenario 5
        elif (p_class == 1):  # delayed (Yellow)
            val = 0.81 / (math.pow((time / 160), 2.41) + 1)  # Scenario 5
        elif (p_class == 2):  # Green
            val = 1.0
            # val = 0.0
        elif (p_class == 3): # Black
            val = 0.0
        # else:
        #     print(self.state, p_class)
        return val


    # def set_seed(self, seed):
    #     # Fix initial seed
    #     super().reset(seed)

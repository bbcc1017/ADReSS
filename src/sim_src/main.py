import argparse
import yaml
import os
import random
import numpy as np
import time
from scipy.stats import t

from ScenarioManager import ScenarioManager
from RuleManager import RuleManager
from MCIEnvironment_gymnasium import MCIEnvironment_gym

# Create and register argument parser
parser = argparse.ArgumentParser(description='Run MCI_simulation')
parser.add_argument('--config_path', default="./config.yaml", help='path to configuration file(.yaml)')
args = parser.parse_args()


class RunManager():
    def __init__(self, args):
        # 0. Load YAML file from configuration file(.yaml) path
        config_path = args.config_path
        with open(config_path, 'r', encoding="utf-8") as f:
            configs = yaml.safe_load(f)

        # 1. Run setting
        cfg_run = configs['run_setting']
        totalSamples = cfg_run['totalSamples']  # number of samples
        self.init_random_seed = cfg_run['random_seed']
        if self.init_random_seed is not None:
            random.seed(self.init_random_seed) # fix python random seed
            np.random.seed(self.init_random_seed) # fix numpy random seed
            rng = np.random.default_rng(self.init_random_seed) # fix and use numpy random number generator seed
        else:
            rng = np.random.default_rng()

        output_path = cfg_run['output_path']
        exp_indicator = cfg_run['exp_indicator']
        output_path = os.path.join(output_path, exp_indicator)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True) # allow overwrite
        save_info = cfg_run['save_info']
        rule_test = cfg_run['rule_test']
        eval_mode = cfg_run['eval_mode']

        # 1. Read data and create scenario
        self.s_manager = ScenarioManager(configs, rng=rng)
        scenario = self.s_manager.scenario
        
        # 2. Create rules
        # configs: rule_name
        self.r_manager = RuleManager(configs['rule_info'], scenario=scenario, rng=rng) # returns collection of Rule objects?
        self.rules = self.r_manager.rules

        # 4. Create environment
        self.env = MCIEnvironment_gym(scenario=scenario,
                                      rng=rng,
                                      rule_test=rule_test,
                                      eval_mode=eval_mode)

        # 5. Run simulation
        output, output_stat = self.run(self.env, self.rules, totalSamples)
        np.savetxt(os.path.join(output_path, "results_{0}.txt".format(exp_indicator)), output, fmt='%s', delimiter="  ")
        np.savetxt(os.path.join(output_path, "results_{0}_stat.txt".format(exp_indicator)), output_stat, fmt='%s', delimiter="  ")

    def set_random_seed(self, seed):
        seed += self.init_random_seed
        random.seed(seed)  # fix python random seed
        np.random.seed(seed)  # fix numpy random seed
        rng = np.random.default_rng(seed)  # fix and use numpy random number generator seed

        self.s_manager.set_seed(rng) # self.s_manager.scenario.ev_manager.set_seed(rng)
        for r in self.rules:
            r.set_seed(rng)
        self.env.set_seed(rng)

    def run(self, env, rules, totalSamples):
        def safe_pdr(saved_reward, preventable_reward):
            if preventable_reward <= 0:
                return 0.0
            return 1 - saved_reward / preventable_reward

        results_rew = np.zeros((len(rules), totalSamples), dtype=float)
        results_time = np.zeros((len(rules), totalSamples), dtype=float)
        results_pdr = np.zeros((len(rules), totalSamples), dtype=float)
        results_rewWOG = np.zeros((len(rules), totalSamples), dtype=float)
        results_pdrWOG = np.zeros((len(rules), totalSamples), dtype=float)
        # results_preventable = np.zeros((len(rules), totalSamples), dtype=float)

        rule_names = np.array([rule.rule_name for rule in rules])

        action_logs = []
        for iter in range(1, totalSamples + 1):
            print("Iter :", iter)
            action_log = {'red_uav': 0, 'red_amb': 0, 'yellow_uav': 0, 'yellow_amb': 0, 'green': 0}
            for r_idx in range(len(rules)):
                print(rules[r_idx].rule_name)
                self.set_random_seed(iter)
                obs, _ = env.reset()
                total_Green = (obs['p_states'][:, 0] == 2).sum()
                done = False
                cumul_reward = 0
                count = 0
                while not done:
                    action = rules[r_idx].select(obs)
                    # print(env.state)
                    # print(action)
                    # print(obs['num_amb'], obs['num_uav'])
                    # if obs['num_amb'] > 0 and obs['num_uav'] > 0:
                    #     print("Observation: ", obs, "Action: ", action)

                    if action[1] != 0:  # patient transport case (not staying at scene or returning)
                        pat_level = action[0]
                        trans_veh = action[2]
                        if pat_level == 0:  # red
                            if trans_veh == 0:  # amb
                                action_log['red_amb'] += 1
                            else:
                                action_log['red_uav'] += 1
                        elif pat_level == 1:  # yellow
                            if trans_veh == 0:  # amb
                                action_log['yellow_amb'] += 1
                            else:
                                action_log['yellow_uav'] += 1
                        else:  # green
                            action_log['green'] += 1
                    # action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)

                    cumul_reward += reward
                    if (r_idx == 0) & (reward < 0.0):
                        print('now')
                    # print(obs['num_amb'],obs['num_uav'])
                    # print("Survival rate sum at this decision point: ", reward, "Current time", info['time'])
                action_logs.append(action_log)
                # print("Total survival rate sum: ", cumul_reward, "MCI end time", info['time'])
                # print("Simulation {}-{} finished".format(iter,r_idx))
                results_rew[r_idx, iter - 1] = cumul_reward
                results_time[r_idx, iter - 1] = info['time']
                results_pdr[r_idx, iter - 1] = safe_pdr(cumul_reward, env.preventable)
                results_rewWOG[r_idx, iter - 1] = cumul_reward - total_Green
                results_pdrWOG[r_idx, iter - 1] = safe_pdr(cumul_reward - total_Green,
                                                           env.preventable - total_Green)
                # results_preventable[r_idx, iter - 1] = env.preventable

        stat_rew = np.zeros((len(rules), 3), dtype=float)
        stat_time = np.zeros((len(rules), 3), dtype=float)
        stat_pdr = np.zeros((len(rules), 3), dtype=float)
        stat_rewWOG = np.zeros((len(rules), 3), dtype=float)
        stat_pdrWOG = np.zeros((len(rules), 3), dtype=float)
        def get_CI(data):
            n_sample = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            std_err = std / (n_sample ** 0.5)
            CI = t.interval(confidence=0.95, df = n_sample-1, loc = mean, scale = std_err)
            return mean, std, (CI[1] - CI[0])/2

        for r_idx in range(len(rules)):
            stat_rew[r_idx][:] = get_CI(results_rew[r_idx])
            stat_time[r_idx][:] = get_CI(results_time[r_idx])
            stat_pdr[r_idx][:] = get_CI(results_pdr[r_idx])
            stat_rewWOG[r_idx][:] = get_CI(results_rewWOG[r_idx])
            stat_pdrWOG[r_idx][:] = get_CI(results_pdrWOG[r_idx])
        stat_print_rew = np.column_stack((rule_names, stat_rew))
        stat_print_time = np.column_stack((rule_names, stat_time))
        stat_print_pdr = np.column_stack((rule_names, stat_pdr))
        stat_print_rewWOG = np.column_stack((rule_names, stat_rewWOG))
        stat_print_pdrWOG = np.column_stack((rule_names, stat_pdrWOG))

        outcomes_rew = np.column_stack((rule_names, results_rew))
        outcomes_time = np.column_stack((rule_names, results_time))
        outcomes_pdr = np.column_stack((rule_names, results_pdr))
        outcomes_rewWOG = np.column_stack((rule_names, results_rewWOG))
        outcomes_pdrWOG = np.column_stack((rule_names, results_pdrWOG))
        # outcomes_preventable = np.column_stack((rule_names, results_preventable))

        output = np.concatenate((outcomes_rew, outcomes_time, outcomes_pdr, outcomes_rewWOG, outcomes_pdrWOG), axis=0)
        output_stat = np.concatenate((stat_print_rew, stat_print_time, stat_print_pdr, stat_print_rewWOG, stat_print_pdrWOG), axis=0)

        return output, output_stat

if __name__ == '__main__':
    start_t = time.time()
    run_model = RunManager(args)
    print(f"Computation time(s): {time.time() - start_t}")

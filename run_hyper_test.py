import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_script(params):
    dataset, ew, es, b = params
    script_path = "asml_hyper_reg_run.py"
    command = [
        "python", script_path,
        "--dataset", dataset,
        "--exploration_window", str(ew),
        "--ensemble_size", str(es),
        "--budget", str(b),
    ]
    subprocess.run(command)

# List of dataset names
dataset_name_list = [
    # 'ailerons',
    # 'elevators',
    # 'fried',
    # 'hyperA',
    # 'FriedmanGsg',
    'FriedmanGra',
    'FriedmanLea',
    # 'kin8nm',
    # 'abalone',
    # 'bike',
    # 'House8L',
    # 'MetroTraffic',
    'cpu_activity',
    'white_wine',
]

# Default values
default_ensemble_size = 5
default_exploration_window = 1000
default_budget = 10
default_prediction_mode = 'ensemble'

exploration_window = [100, 1000, 3000, 5000, 7000, 10000]
budget = [5, 10, 15, 20, 25]
ensemble_size = [3, 5, 7, 9, 11]


# # Run the script with different parameters
# with ThreadPoolExecutor() as executor:
#     for dataset in dataset_name_list:
#         # Vary only exploration_window
#         params_list = [(dataset, ew, default_ensemble_size, default_budget) for ew in exploration_window]
#         executor.map(run_script, params_list)
    
#     for dataset in dataset_name_list:
#         # Vary only ensemble_size
#         params_list = [(dataset, default_exploration_window, es, default_budget) for es in ensemble_size]
#         executor.map(run_script, params_list)
    
#     for dataset in dataset_name_list:
#         # Vary only budget
#         params_list = [(dataset, default_exploration_window, default_ensemble_size, b) for b in budget]
#         executor.map(run_script, params_list)

with ThreadPoolExecutor() as executor:
    
    params_list = []
    
    for dataset in dataset_name_list:
        # Vary only exploration_window
        params_list_ew = [(dataset, ew, default_ensemble_size, default_budget) for ew in exploration_window]
        params_list.extend(params_list_ew)
        # Vary only ensemble_size
        params_list_es = [(dataset, default_exploration_window, es, default_budget) for es in ensemble_size]
        params_list.extend(params_list_es)
        # Vary only budget
        params_list_b = [(dataset, default_exploration_window, default_ensemble_size, b) for b in budget]
        params_list.extend(params_list_b)
    
    executor.map(run_script, list(set(params_list)))    
        

from capymoa.regressor import SOKNL

import psutil
import time
import json
import random

from capymoa.stream import stream_from_file
from capymoa.evaluation import RegressionEvaluator, RegressionWindowedEvaluator

import warnings
warnings.filterwarnings("ignore")

import argparse

def main(dataset_name:str,run_count:int=None,seed:int=42):
    
    #seed = random.randint(42,52)
    
    print(f"Loading dataset: {dataset_name}, Run Count: {run_count}, Random Seed:{seed}")

    stream = stream_from_file(f"RDatasets/{dataset_name}.arff")

    regressionEvaluator = RegressionEvaluator(schema=stream.get_schema())
    regressionWindowedEvaluator = RegressionWindowedEvaluator(schema=stream.get_schema(),window_size=1000)

    soknl = SOKNL(schema=stream.get_schema(),random_seed=seed)

    t=0
    times = []
    memories = []
    while stream.has_more_instances():
        instance = stream.next_instance()
        mem_before = psutil.Process().memory_info().rss # Recording Memory
        start = time.time()  # Recording Time
        try:
            prediction = soknl.predict(instance)
        except Exception as e:
            print(f"Error while prediction: {e}")
            prediction = 0.0
        #print(f"y_true: {instance.y_value}, y_pred: {prediction}")
        regressionEvaluator.update(instance.y_value, prediction)
        regressionWindowedEvaluator.update(instance.y_value, prediction)
        try:
            soknl.train(instance)
        except Exception as e:
            print(f"Error while learning: {e}")
        end = time.time()
        mem_after = psutil.Process().memory_info().rss
        iteration_mem = mem_after - mem_before
        memories.append(iteration_mem)
        iteration_time = end - start
        times.append(iteration_time)
        t+=1
        if t%1000==0:
            print(f"Running Instance **{t}**")
            print(f"R2 score - {round(regressionEvaluator.R2(),3)}")
            print(f"RMSE score - {round(regressionEvaluator.RMSE(),3)}")
            print("-"*40)

    # saving results in dict
    save_record = {
        "model": 'SOKNL',
        "dataset": dataset_name,
        "regressionEvaluator": regressionEvaluator.metrics_dict(),
        "windows_scores": regressionWindowedEvaluator.metrics_per_window().to_dict(orient='list'),
        "time": times,
        "memory": memories
    }

    if run_count is not None:
        file_name = f"{save_record['model']}_{save_record['dataset']}_{run_count}.json"
    else:
        file_name = f"{save_record['model']}_{save_record['dataset']}.json"

    # To store the dictionary in a JSON file
    with open(f"TEMP/{file_name}", 'w') as json_file:  # change temp to  saved_results_json for final run
        json.dump(save_record, json_file)

    print(f"Results saved successfully in TEMP/{file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SOKNL on a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--run_count', type=int, default=None, help='Run count')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    args = parser.parse_args()
    main(dataset_name=args.dataset,
         run_count=args.run_count,
         seed=args.seed)
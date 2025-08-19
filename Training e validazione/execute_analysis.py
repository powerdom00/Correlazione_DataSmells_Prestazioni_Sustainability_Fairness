import os

from adult import adult_standard_models
from adult import adult_preprocess_techniques
from adult import adult_inprocess_techniques

import time

def calculate_fibonacci(n): 
    if n <= 1: 
        return n 
    else: 
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2) 
    
def warm_up_cpu(duration_minutes=2): 
    end_time = time.time() + 60 * duration_minutes 
    fib_number = 30 # A relatively high number to ensure CPU intensity, adjust as needed 
    print(f"Starting CPU warm-up by calculating Fibonacci of {fib_number} repeatedly for {duration_minutes} minute(s)...") 
    while time.time() < end_time: 
        result = calculate_fibonacci(fib_number) 
        print(f"Fibonacci calculation result: {result}") 
        print("CPU warm-up complete.") 
        


def run_adult_dataset_analysis():
 
    adult_standard_models.run_model_training()
    
    os.replace("emissions.csv", "src/output/reports/adult/emissions(in).csv")
    

if __name__=="__main__":
    
    warm_up_cpu(duration_minutes=2)
    
    run_adult_dataset_analysis()

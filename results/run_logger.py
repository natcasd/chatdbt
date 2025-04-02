import os
import csv
from datetime import datetime

def log_run_results(results, log_file="results/experiment_results.csv"):
    """
    Log the results of a model run to a CSV file.
    
    Args:
        results (dict): Dictionary containing run results with the following keys:
            - time_elapsed (float): Time taken for the run in seconds
            - dataset (str): Dataset used for the run
            - pred (list): Predicted values
            - true (list): True values
            - provider (str): Provider used (OpenAI, Anthropic, Groq, etc.)
            - model (str): Model used
            - approach (str): Approach or method used
            - accuracy (float): Accuracy score
            - precision (float): Precision score
            - recall (float): Recall score
            - f1 (float): F1 score
        log_file (str): Path to the CSV log file
    """
    # Add current date and time automatically
    results['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define the fields to be stored in the CSV
    fields = ['approach', 'dataset', 'pred', 'true', 'accuracy', 'precision', 'recall', 'f1', 'time_elapsed', 'date', 'model', 'provider' ]
    
    # Check if file exists to determine if headers need to be written
    file_exists = os.path.isfile(log_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Write results to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write only the specified fields
        row = {field: results.get(field, '') for field in fields}
        writer.writerow(row)
    
    print(f"Run results logged to {log_file}")

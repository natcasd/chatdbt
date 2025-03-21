from metrics.metrics import accuracy, precision, recall, f1_score
from run_logger import log_run_results
import time
from tqdm import tqdm

def approach1(records, model_client, dataset_name="not defined", log_results=False, prompt_template=None, system_prompt=None):
    pred = []
    true = []
    
    start_time = time.time()

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that strictly answers True or False based on patient data and provided semantic regex matching."

    for record in tqdm(records, desc="Processing records", unit="record"):
        regex = record['s_regex']
        record_text = record['record']
        label  = record['match']
        
        # Use the provided prompt template if available, otherwise use the default
        if prompt_template:
            prompt = prompt_template.format(regex=regex, record_text=record_text)
        else:
            prompt = f"Given the following patient record, identify if it matches the following semantic regex pattern: {regex} Return either true OR false, nothing else.\n\nPatient Record: {record_text}"
        response = model_client.generate(prompt, system_prompt=system_prompt)
        response_bool = response.strip().lower() == "true"
        pred.append(response_bool)
        true.append(label)

    end_time = time.time()

    acc = accuracy(pred, true)
    prec = precision(pred, true)
    rec = recall(pred, true)
    f1 = f1_score(pred, true)
    
    print(f'accuracy of model over {len(records)} generated records: {acc}')
    print(f'precision of model over {len(records)} generated records: {prec}')
    print(f'recall of model over {len(records)} generated records: {rec}')
    print(f'f1 of model over {len(records)} generated records: {f1}')

    results = {
        'approach': 'Approach 1',
        'dataset': dataset_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'time_elapsed': end_time - start_time,
        'model': model_client.model,
        'provider': model_client.provider
    }
    if log_results:
        log_run_results(results)
    return pred, true
from metrics.metrics import accuracy, precision, recall, f1_score
from results.run_logger import log_run_results
import time
from tqdm import tqdm

def approach1(records, model_client, dataset_name="not defined", log_results=False, prompt_template=None, system_prompt=None, verbose=False):
    pred = []
    true = []
    responses = []
    start_time = time.time()

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that strictly answers True or False based on patient data and provided semantic regex matching."

    # Use tqdm only when not in verbose mode
    if verbose:
        record_iterator = records
        print(f"Processing {len(records)} records...")
    else:
        record_iterator = tqdm(records, desc="Processing records", unit="record")

    for record in record_iterator:
        regex = record['s_regex']
        record_text = record['record']
        label  = record['match']
        
        # Use the provided prompt template if available, otherwise use the default
        if prompt_template:
            prompt = prompt_template.format(regex=regex, record_text=record_text)
        else:
            prompt = f"Given the following patient record, identify if it matches the following semantic regex pattern: {regex} Return either true OR false, nothing else.\n\nPatient Record: {record_text}"
        response = model_client.generate(prompt=prompt, system_prompt=system_prompt)
        if isinstance(response, str):
            response_bool = response.strip().lower() == "true"
        else:
            response_bool = response
        responses.append(response)
        pred.append(response_bool)
        true.append(label)
        
        if verbose:
            print("\n" + "="*80)
            print(f"\nPatient Record: {record_text}")
            print(f"Semantic Regex: {regex}")
            print(f"Model prediction: {response_bool}, Actual: {label}")
            print("=" * 80)

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
        'pred': pred,
        'true': true,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'time_elapsed': end_time - start_time,
        'model': model_client.model,
        'provider': model_client.provider,
        'responses': responses 
    }
    if log_results:
        log_run_results(results)
    return results

def approach1_naive(records, model_client, dataset_name="not defined", log_results=False, prompt_template=None, system_prompt=None, verbose=False):
    pred = []
    true = []
    responses = []
    start_time = time.time()

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that strictly answers True or False based on patient data and provided natural language query."

    # Use tqdm only when not in verbose mode
    if verbose:
        record_iterator = records
        print(f"Processing {len(records)} records...")
    else:
        record_iterator = tqdm(records, desc="Processing records", unit="record")

    for record in record_iterator:
        nl_query = record['nl_query']
        record_text = record['record']
        label  = record['match']
        
        # Use the provided prompt template if available, otherwise use the default
        if prompt_template:
            prompt = prompt_template.format(nl_query=nl_query, record_text=record_text)
        else:
            prompt = f"Given the following patient record, answer the following natural language query: {nl_query} Return either true OR false, nothing else.\n\nPatient Record: {record_text}"
        response = model_client.generate(prompt, system_prompt=system_prompt)
        if isinstance(response, str):
            response_bool = response.strip().lower() == "true"
        else:
            response_bool = response
        responses.append(response)
        pred.append(response_bool)
        true.append(label)
        
        if verbose:
            print("\n" + "="*80)
            print(f"\nPatient Record:\n{record_text}")
            print(f"\nSemantic regex query:\n{record['s_regex']}")
            print(f"\nNatural language query:\n{nl_query}")
            print(f"\nModel prediction: {response_bool}, Actual: {label}")
            print("=" * 80)

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
        'approach': 'Approach 1 Naive',
        'dataset': dataset_name,
        'pred': pred,
        'true': true,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'time_elapsed': end_time - start_time,
        'model': model_client.model,
        'provider': model_client.provider,
        'responses': responses
    }
    if log_results:
        log_run_results(results)
    return results
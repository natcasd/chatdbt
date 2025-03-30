from tqdm import tqdm
import ast
import re
import time
from run_logger import log_run_results
from metrics.metrics import accuracy, precision, recall, f1_score 
from transitions import Machine

class PatternFSM:
    def __init__(self, pattern):
        self.pattern = pattern
        self.states = ["START", "EXTRACTING", "ACCEPTED", "REJECTED"]
        self.machine = Machine(model=self, states=self.states, initial="START")

        # Define transitions
        self.machine.add_transition(trigger="start_extraction", source="START", dest="EXTRACTING")
        self.machine.add_transition(trigger="found_symbol", source="EXTRACTING", dest="EXTRACTING")
        self.machine.add_transition(trigger="accept", source="EXTRACTING", dest="ACCEPTED")
        self.machine.add_transition(trigger="reject", source="EXTRACTING", dest="REJECTED")

    def match(self, extracted_symbols, semantic_symbols, verbose=False):
        """ Checks if all required semantic symbols are found in the given text. """
        self.start_extraction() 

        # Extract semantic symbols from the pattern string (e.g., "<sym1><sym2>")
        required_symbols = re.findall(r"<(.*?)>", semantic_symbols)

        found_all = set(required_symbols).issubset(set(extracted_symbols))

        if found_all:
            self.accept()
            if verbose:
                print("FSM Result: All required symbols found.")
            
            return True
        else:
            self.reject()
            missing = set(semantic_symbols) - set(extracted_symbols)

            if verbose:
                print(f"FSM Result: Some required symbols are missing: {missing}")
           
            return False
    # def extract_symbols(self, record_text, semantic_symbols):
    #     found_symbols = set()

    #     for symbol in semantic_symbols:
    #         pattern = re.escape(symbol)  
    #         match = re.search(pattern, record_text, re.IGNORECASE)
            
    #         if match:
    #             found_symbols.add(symbol)
    #             self.found_symbol()  
        
    #     return set(semantic_symbols).issubset(found_symbols)

# want this to return true if pattern exists, false otherwise
# response_dict is a dictionary of <symbol>: extracted text pairs, can adjust what this looks like if needed
def pattern_identification(response_dict, regex, verbose=False):
    fsm = PatternFSM(regex)
    extracted_symbols = list(response_dict.keys())
    return fsm.match(extracted_symbols, regex, verbose)

def parse_model_output(model_output):
    """
    This function takes a model output string and attempts to extract a dictionary from it if it exists.
    
    Args:
        model_output (str): The output string from the model which may contain a dictionary.
        
    Returns:
        dict: A dictionary extracted from the model output if it exists, otherwise an empty dictionary.
    """
    try:
        # Try to find dictionary pattern using regex
        import re
        dict_match = re.search(r'\{.*\}', model_output, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            response_dict = ast.literal_eval(dict_str)
        else:
            # Fallback to original response if no dictionary pattern found
            response_dict = ast.literal_eval(model_output)
    except:
        print(f"Failed to parse response as dictionary: {model_output}")
        response_dict = {}
    return response_dict

def approach2(records, model_client, dataset_name="not defined", log_results=False, extraction_prompt_template=None, system_prompt=None, verbose=False):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that strictly outputs a python dictionary with <symbol>: extracted text pairs, that can be parsed with ast.literal_eval()."

    # Use tqdm only when not in verbose mode
    if verbose:
        record_iterator = records
        print(f"Processing {len(records)} records...")

    else:
        record_iterator = tqdm(records, desc="Processing records", unit="record")

    for record in record_iterator:

        regex = record['s_regex']
        record_text = record['record']
        label = record['match']
 
        if verbose:
            print("\n" + "="*80)
            print(f"Patient Record:\n{record_text}\n", flush=True)
            print(f"Semantic Regex: {regex}")


        if extraction_prompt_template:
            prompt = extraction_prompt_template.format(regex=regex, record_text=record_text)
        else:
            prompt = f"Given the following patient record, extract the following semantic symbols if they exist: {regex}. Return a machine parseable dictionary with <symbol>: extracted text pairs. IMPORTANT: Only return the dictionary, nothing else.\n\nPatient Record: {record_text}"
        
        response = model_client.generate(prompt, system_prompt=system_prompt)
        response_dict = parse_model_output(response)

        
        if verbose:
            print(f"\nExtracted Symbols Dictionary:")
            for symbol, text in response_dict.items():
                print(f"  - {symbol}: {text}")
        
        response_bool = pattern_identification(response_dict, regex, verbose)
        pred.append(response_bool)
        true.append(label)
        
        if verbose:
            print(f"Model prediction: {response_bool}, Actual: {label}")
            print("="*80)

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
        'approach': 'Approach 2',
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
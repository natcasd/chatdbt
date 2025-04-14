from tqdm import tqdm
import ast
import re
import time
from results.run_logger import log_run_results
from metrics.metrics import accuracy, precision, recall, f1_score 
from transitions import Machine

class PatternFSM:
    def __init__(self, pattern, order_sensitive=None):
        if order_sensitive is None:
            order_sensitive = input("Should the pattern matching be order-sensitive? (yes/no): ").strip().lower() == "yes"
        self.pattern = pattern
        self.order_sensitive = order_sensitive
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
            if self.order_sensitive:
                if required_symbols == extracted_symbols:
                    self.accept()
                    if verbose:
                        print("FSM Result: All required symbols found. Correct order.")
                else:
                    self.reject()
                    print("FSM Result: All required symbols found. Incorrect order.")
            else:   
                self.accept()
                if verbose:
                    print("FSM Result: All required symbols found.")
            
            return True
        else:
            self.reject()
            missing = set(required_symbols) - set(extracted_symbols)
            if verbose:
                print(f"FSM Result: Some required symbols are missing: {missing}")
           
            return False

# want this to return true if pattern exists, false otherwise
# response_dict is a dictionary of <symbol>: extracted text pairs, can adjust what this looks like if needed
def pattern_identification(extracted_symbols, regex, verbose=False):
    fsm = PatternFSM(regex)
    # Extract just the symbols from the tuples for pattern matching
    symbols_only = [symbol for symbol in extracted_symbols]
    return fsm.match(symbols_only, regex, verbose)

def parse_model_output(model_output):
    """
    This function takes a model output string and attempts to extract a list of tuples from it if it exists.
    Each tuple should be (symbol, explanation) where symbol is the semantic symbol and explanation is where it appears.
    
    Args:
        model_output (str): The output string from the model which may contain a list of tuples.
        
    Returns:
        list: A list of tuples extracted from the model output if it exists, otherwise an empty list.
    """
    try:
        # Try to find list pattern using regex
        import re
        list_match = re.search(r'\[.*\]', model_output, re.DOTALL)
        if list_match:
            list_str = list_match.group(0)
            response_list = ast.literal_eval(list_str)
        else:
            # Fallback to original response if no list pattern found
            response_list = ast.literal_eval(model_output)
    except:
        print(f"Failed to parse response as list of tuples: {model_output}")
        response_list = []
    return response_list

def approach2(records, model_client, dataset_name="not defined", log_results=False, extraction_prompt_template=None, system_prompt=None, verbose=False):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:

        system_prompt = """
        You are a helpful AI assistant that strictly outputs a python list of tuples, where each tuple is (<semantic_symbol>, explanation) in the order they appear in the patient record.
        The output should be parseable with ast.literal_eval().
        """
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
            print(f"Patient Record:\n{record_text}\n")
            print(f"Semantic Regex: {regex}")
            
        if extraction_prompt_template:
            prompt = extraction_prompt_template.format(regex=regex, record_text=record_text)
        else:
            prompt = """
            Given the following patient record, extract the following semantic symbols if they exist: {regex}. 
            Return a machine parseable python list of tuples, where each tuple is (<semantic_symbol>, explanation) in the order they appear in the patient record. 
            IMPORTANT: Only return the list, nothing else. If a semantic symbol is not clearly represented in the patient record, do not include it in the list.
            Make sure the order of the list reflects the order in which the semantic symbols appear in the patient record, not the order in which they are listed in the regex.
            The explanation should be a brief description of where/how the symbol appears in the text.
            \n\nPatient Record: {record_text}
            """
        
        response = model_client.generate(prompt, system_prompt=system_prompt)
        extracted_symbols = parse_model_output(response)
        
        if verbose:
            print(f"\nExtracted Symbols List:")
            #for symbol, explanation in extracted_symbols:
                #print(f"  - {symbol}: {explanation}")
            for item in extracted_symbols:
                print(f" - {item}")
        
        response_bool = pattern_identification(extracted_symbols, regex, verbose)
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
        'pred': pred,
        'true': true,
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
    return results
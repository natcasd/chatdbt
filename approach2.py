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
        self.machine.add_transition(trigger="accept", source="EXTRACTING", dest="ACCEPTED")
        self.machine.add_transition(trigger="reject", source="EXTRACTING", dest="REJECTED")

    def match(self, record_text):
        # Move to extracting state
        self.start_extraction()  

        # Extract text values
        extracted_symbols = self.extract_symbols(record_text)  

        if extracted_symbols:
            # If we found all required symbols, accept
            self.accept()  
            print("Extracted Symbols:", extracted_symbols)
            return extracted_symbols
        else:
            # If symbols are missing, reject
            self.reject()  
            # print("No valid symbols found.")
            return None
    def extract_symbols(self, record_text):
        return None

# want this to return true if pattern exists, false otherwise
# response_dict is a dictionary of <symbol>: extracted text pairs, can adjust what this looks like if needed
def pattern_identification(symbols, regex):
    fsm = PatternFSM(re.compile(regex))
    fsm.match(symbols)
    return False

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

def approach2(records, model_client, dataset_name="not defined", log_results=False, extraction_prompt_template=None, system_prompt=None):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that strictly outputs a python dictionary with <symbol>: extracted text pairs, that can be parsed with ast.literal_eval()."

    for record in tqdm(records, desc="Processing records", unit="record"):
        regex = record['s_regex']
        record_text = record['record']
        label  = record['match']
        if extraction_prompt_template:
            prompt = extraction_prompt_template.format(regex=regex, record_text=record_text)
        else:
            prompt = f"Given the following patient record, extract the following semantic symbols if they exist: {regex}. Return a machine parseable dictionary with <symbol>: extracted text pairs. IMPORTANT: Only return the dictionary, nothing else.\n\nPatient Record: {record_text}"
            
        response = model_client.generate(prompt, system_prompt=system_prompt)
        # this doesn't work unless llm returns exactly the dictionary - perhaps some parsing logic would be better
        response_dict = parse_model_output(response)

        response_bool = pattern_identification(response_dict, regex)
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
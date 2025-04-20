from tqdm import tqdm
import re
import time
from results.run_logger import log_run_results
from metrics.metrics import accuracy, precision, recall, f1_score 
from transitions import Machine
from approaches.helper_methods import extract_list_from_model_output, extract_symbols_from_annotated_record

class PatternFSM:
    def __init__(self, pattern, order_sensitivity=None):
        # self.pattern = pattern
        # self.order_sensitive = order_sensitive
        # self.states = ["START", "EXTRACTING", "ACCEPTED", "REJECTED"]
        # self.machine = Machine(model=self, states=self.states, initial="START")

        # # Define transitions
        # self.machine.add_transition(trigger="start_extraction", source="START", dest="EXTRACTING")
        # self.machine.add_transition(trigger="found_symbol", source="EXTRACTING", dest="EXTRACTING")
        # self.machine.add_transition(trigger="accept", source="EXTRACTING", dest="ACCEPTED")
        # self.machine.add_transition(trigger="reject", source="EXTRACTING", dest="REJECTED")
         ######## Non-native FSM implementation ########
        # Extract required semantic symbols from something like "<A><B><C>"
        self.symbol_sequence = re.findall(r"<(.*?)>", pattern)
        self.state_names = [f"STATE_{i}" for i in range(len(self.symbol_sequence) + 1)]

        # Create machine with states from STATE_0 to STATE_n (accepting)
        self.machine = Machine(model=self, states=self.state_names, initial=self.state_names[0])

        # Add transitions for each expected symbol: STATE_i --symbol--> STATE_{i+1}
        for i, symbol in enumerate(self.symbol_sequence):
            self.machine.add_transition(trigger=symbol, source=self.state_names[i], dest=self.state_names[i + 1])

        self.final_state = self.state_names[-1]

    def set_match(self, extracted_symbols, semantic_symbols, verbose):
        """
        Checks if all required semantic symbols (from the semantic_pattern) are present
        in the extracted symbols, regardless of order.
        """
        # Extract symbol names from pattern like "<A><B><C>" → ['A', 'B', 'C']
        required_symbols = set(re.findall(r"<(.*?)>", semantic_pattern))

        # Pull just the symbol names from extracted tuples and clean them
        extracted_symbols = set(symbol.strip("<>") for symbol, _ in extracted_symbol_tuples)

        if verbose:
            print(f"Required symbols: {required_symbols}")
            print(f"Extracted symbols: {extracted_symbols}")

        return required_symbols.issubset(extracted_symbols)

    def fsm_match(self, extracted_symbols, semantic_symbols, verbose):
        expected_index = 0  # track which symbol we're looking for next

        # Extract only the first element of each tuple for symbol matching
        flattened_symbols = [symbol.strip("<>") for symbol, _ in extracted_symbols]

        if verbose:
            print(f"Required sequence: {self.symbol_sequence}")
            print(f"Extracted sequence: {flattened_symbols}\n")

        for symbol in flattened_symbols:
            if expected_index < len(self.symbol_sequence):
                expected_symbol = self.symbol_sequence[expected_index]
                if symbol == expected_symbol:
                    # Perform the transition using symbol name as trigger
                    if hasattr(self, symbol):
                        getattr(self, symbol)()
                        expected_index += 1
                        if verbose:
                            print(f"Matched '{symbol}' → transitioned to {self.state}")

        accepted = self.state == self.final_state
        if verbose:
            print(f"\nFinal state: {self.state} → {'ACCEPTED' if accepted else 'REJECTED'}")

        return accepted

# want this to return true if pattern exists, false otherwise
# extracted_symbols is a list of tuples of (<symbol>: explanation), can adjust what this looks like if needed
def pattern_identification(extracted_symbols, regex, order_sensitivity, verbose):
    # making it so its case insensitive
    extracted_symbols = [(symbol.lower(), explanation) for symbol, explanation in extracted_symbols]
    regex = regex.lower()
    fsm = PatternFSM(regex, order_sensitive=order_sensitivity)

    if order_sensitivity:
        # Order-sensitive FSM match (e.g., requires A→B→C)
        return fsm.fsm_match(extracted_symbols, regex, verbose)
    else:
        # Order-insensitive: just check if all required symbols are present
        symbols_only = [symbol.strip("<>") for symbol, _ in extracted_symbols]
        required_symbols = set(re.findall(r"<(.*?)>", regex))

        if verbose:
            print(f"Required symbols (unordered): {required_symbols}")
            print(f"Extracted symbols: {symbols_only}")

        return required_symbols.issubset(set(symbols_only))

def approach2_base(records, model_client, dataset_name="not defined", log_results=False, extraction_prompt_template=None, system_prompt=None, verbose=False, order_sensitive=False):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:

        system_prompt = """
        You are a helpful AI assistant that strictly outputs a python list of tuples, where each tuple is ("semantic_symbol", "explanation") in the order they appear in the patient record.
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
            prompt = extraction_prompt_template
        else:
            prompt = """
            Given the following patient record, extract the following semantic symbols if they exist: {regex}. 
            Return a machine parseable python list of tuples, where each tuple is ("semantic_symbol", "explanation") in the order they appear in the patient record. Only include semantic symbols that are explicitly represented in the patient record, i.e. their explanation should be the actual text in which they appear. If they don't appear don't include them in the list. 
            IMPORTANT: Only return the list, nothing else.
            The semantic symbols should just be the word not the <>
            Make sure the order of the list reflects the order in which the semantic symbols appear in the patient record, not the order in which they are listed in the regex.
            The explanation should be a brief description of where/how the symbol appears in the text.
            \n\nPatient Record: {record_text}
            """

        prompt = prompt.format(regex=regex, record_text=record_text)
        response = model_client.generate(prompt, system_prompt=system_prompt)
        extracted_symbols = extract_list_from_model_output(response)

        if verbose:
            print(f"\nExtracted Symbols List:")
            for symbol, explanation in extracted_symbols:
                print(f"  - {symbol}: {explanation}")
        
        response_bool = pattern_identification(extracted_symbols, regex, order_sensitive, verbose)
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
        'approach': 'Approach 2 base',
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

def approach2_annotate(records, model_client, dataset_name="not defined", log_results=False, annotation_prompt_template=None, extraction_prompt_template=None, system_prompt=None, manual_extraction=True, verbose=False, order_sensitive=False):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:

        system_prompt = """
        You are an expert medical annotation assistant. Given a patient record, you identify and annotate full contextual text spans corresponding exactly to provided semantic symbols.

        Annotate by surrounding each identified context with tags as:
        <semantic_symbol> identified context text </semantic_symbol>

        Only annotate contexts representing the listed semantic symbols. Return the entire annotated patient record unaltered, preserving all original formatting and punctuation.
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
            
        if annotation_prompt_template:
            annotation_prompt = annotation_prompt_template
        else:
            annotation_prompt = """
            Task:
            Annotate the patient record by identifying entire contextual spans that correspond to the provided semantic symbols. Each semantic symbol should encompass the full relevant phrase or sentence, not just an isolated word.

            Guidelines:
            - Each semantic symbol must be annotated using tags in this format:
            <semantic_symbol>full contextual text span</semantic_symbol>
            - Annotate only if the patient record explicitly contains contexts that match the provided semantic symbols.
            - Preserve the exact original formatting, punctuation, capitalization, and spacing.
            - Do not modify the original record's text other than inserting annotations.
            - Make sure to close a tag before starting a new one. Do not have nested/overlapping tags.
            - Do not add any additional text or tags outside of the original record.

            Input:
            Semantic Symbols:
            {regex}

            Patient Record:
            {record_text}

            Output:
            Return only the fully annotated patient record as a single continuous string.
            """
        annotated_prompt = annotation_prompt.format(regex=regex, record_text=record_text)
        annotated_record = model_client.generate(annotated_prompt, system_prompt=system_prompt,  max_tokens=1500)
        if verbose:
            print(f"\nAnnotated Record:")
            print(annotated_record)
        
        if manual_extraction:
            extracted_symbols = extract_symbols_from_annotated_record(annotated_record)
        else:
            if extraction_prompt_template:
                extraction_prompt = extraction_prompt_template
            else:
                extraction_prompt = """
                Extract all text enclosed in semantic tags from the annotated medical record below. 

                Return only the extracted content as a valid Python list of tuples in the format: [("tag", "text"), ...].

                Tags may be nested. In such cases, include all relevant entries separately, even if they overlap.

                If a "text" corresponding to a tag is longer that 50 words or a sentence, just return the first 50 words or sentence.

                Do not generate code to extract the text. Just return the list of tuples.

                Example:
                Input: <patient>The <vaccine>flu vaccine</vaccine> was given.</patient>
                Output: [("vaccine", "flu vaccine"), ("patient", "The flu vaccine was given.")]

                Annotated Medical Record:
                {annotated_record}
                """

            extraction_prompt = extraction_prompt.format(annotated_record=annotated_record)
            extracted_symbols = model_client.generate(extraction_prompt, max_tokens=750)
            extracted_symbols = extract_list_from_model_output(extracted_symbols)

        extracted_symbols = [(symbol.lower(), explanation) for symbol, explanation in extracted_symbols]
        if verbose:
            print(f"\nExtracted Symbols List:")
            for symbol, explanation in extracted_symbols:
                print(f"  - {symbol}: {explanation}")
        
        response_bool = pattern_identification(extracted_symbols, regex, order_sensitive, verbose)
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
        'approach': f"Approach 2 {'manual' if manual_extraction else 'llm'} extraction",
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

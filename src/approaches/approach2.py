from tqdm import tqdm
import re
import time
from results.run_logger import log_run_results
from metrics.metrics import accuracy, precision, recall, f1_score 
from transitions import Machine
from approaches.helper_methods import extract_list_from_model_output, extract_symbols_from_annotated_record

class PatternFSM:
    def __init__(self, pattern, order_sensitive=None):
        self.pattern = pattern
        self.order_sensitive = order_sensitive
        
        # Extract required semantic symbols from something like "<A><B><C>"
        self.symbol_sequence = re.findall(r"<(.*?)>", pattern)
        self.state_names = [f"STATE_{i}" for i in range(len(self.symbol_sequence) + 1)]

        # Create machine with states from STATE_0 to STATE_n (accepting)
        self.machine = Machine(model=self, states=self.state_names, initial=self.state_names[0])

        # Add transitions for each expected symbol: STATE_i --symbol--> STATE_{i+1}
        for i, symbol in enumerate(self.symbol_sequence):
            self.machine.add_transition(trigger=symbol, source=self.state_names[i], dest=self.state_names[i + 1])

        self.final_state = self.state_names[-1]

    # Define the FSM transition function, if a match is found, it will transition to the next state
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
def pattern_identification(extracted_symbols, regex, order_sensitive, verbose):
    fsm = PatternFSM(regex, order_sensitive=order_sensitive)

    if order_sensitive:
        # Order-sensitive FSM match (e.g., requires A→B→C)
        return fsm.fsm_match(extracted_symbols, regex, verbose)
    else:
        # Order-insensitive: just check if all required symbols are present using a set
        symbols_only = [symbol.strip("<>") for symbol, _ in extracted_symbols]
        required_symbols = set(re.findall(r"<(.*?)>", regex))

        if verbose:
            print(f"Required symbols (unordered): {required_symbols}")
            print(f"Extracted symbols: {symbols_only}")

        return required_symbols.issubset(set(symbols_only))


# splitting the input into units for processing
def split_record_into_units(record_text, paragraphs_per_unit=3):
    """
    Splits a medical record into streaming units.
    Priority:
        1. Split by section headers (all caps lines like 'HISTORY OF PRESENT ILLNESS').
        2. Fallback to clustering paragraphs (default 3 paragraphs per unit).

    Args:
        record_text (str): The full medical record text.
        paragraphs_per_unit (int): Number of paragraphs to cluster when falling back.

    Returns:
        list of str: List of streaming units.
    """
    units = []

    # Attempt to split by SECTION HEADERS
    section_pattern = r"(?<=\n)([A-Z\s]{3,50})(?=\n)"
    matches = list(re.finditer(section_pattern, record_text))

    if matches:
        # Section headers detected
        starts = [match.start() for match in matches]
        starts.append(len(record_text))  # add end of record for slicing last section

        for i in range(len(matches)):
            start = starts[i]
            end = starts[i+1]
            section_text = record_text[start:end].strip()
            if section_text:
                units.append(section_text)

    else:
        # Fallback: Split into paragraphs
        paragraphs = [p.strip() for p in record_text.split("\n\n") if p.strip()]
        
        for i in range(0, len(paragraphs), paragraphs_per_unit):
            group = paragraphs[i:i+paragraphs_per_unit]
            unit_text = "\n\n".join(group)
            units.append(unit_text)

    return units


# Define a function to process the record in streaming units, calls LLM per unit and use fsm to check if match is found, and immediately halts if match is found
def streaming_unit_processing(record_text, regex, model_client, system_prompt, order_sensitive, verbose=False):
    """
    Processes a medical record in streaming units (sections first, fallback to paragraph clusters).
    Calls LLM per unit, accumulates extracted symbols, and halts immediately if match is found.

    Args:
        record_text (str): Full medical record text.
        regex (str): Semantic pattern to match (e.g., "<patient><diagnosis>").
        model_client: LLM client instance for generation.
        system_prompt (str): System prompt for LLM extraction.
        order_sensitive (bool): Whether matching requires correct order (FSM) or not.
        verbose (bool): Whether to print detailed logs.

    Returns:
        (bool, list): (matched, extracted_symbols_so_far)
    """
    accumulated_extracted_symbols = []
    units = split_record_into_units(record_text)

    for unit_number, unit_text in enumerate(units, start=1):
        if verbose:
            print(f"\nStreaming Unit {unit_number}:")
            print(unit_text)
            print("-" * 80)

        # Build the LLM prompt for this unit
        prompt = f"""
        Given the following medical text, extract the following semantic symbols if they exist: {regex}. 
        Return a machine-parseable Python list of tuples, where each tuple is (semantic_symbol, explanation).
        Only include symbols that are explicitly represented in the text.
        IMPORTANT: Only return the list, nothing else.

        Text:
        {unit_text}
        """

        # Call LLM
        response = model_client.generate(prompt, system_prompt=system_prompt)
        unit_extracted_symbols = extract_list_from_model_output(response)

        # Normalize symbols (if necessary, e.g., lowercase)
        unit_extracted_symbols = [(symbol.lower().strip("<>"), explanation) for symbol, explanation in unit_extracted_symbols]

        # Accumulate extracted symbols
        accumulated_extracted_symbols.extend(unit_extracted_symbols)

        # Pattern matching check
        match_found = pattern_identification(accumulated_extracted_symbols, regex, order_sensitive, verbose)

        if match_found:
            if verbose:
                print(f"\n✅ Match found at streaming unit {unit_number}!")
            return True, accumulated_extracted_symbols

    # No match after processing all units
    if verbose:
        print("\n❌ No match found after streaming all units.")
    return False, accumulated_extracted_symbols

def approach2_base(records, model_client, dataset_name="not defined", log_results=False, extraction_prompt_template=None, system_prompt=None, verbose=False, order_sensitive=False, streaming=False):
    pred = []
    true = []

    start_time = time.time()

    if system_prompt is None:

        system_prompt = """
        You are a helpful AI assistant that strictly outputs a python list of tuples, where each tuple is (semantic_symbol, explanation) in the order they appear in the patient record.
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
            Return a machine parseable python list of tuples, where each tuple is (semantic_symbol, explanation) in the order they appear in the patient record. Only include semantic symbols that are explicitly represented in the patient record, i.e. their explanation should be the actual text in which they appear. If they don't appear don't include them in the list. 
            IMPORTANT: Only return the list, nothing else.
            The semantic symbols should just be the word not the <>
            Make sure the order of the list reflects the order in which the semantic symbols appear in the patient record, not the order in which they are listed in the regex.
            The explanation should be a brief description of where/how the symbol appears in the text.
            \n\nPatient Record: {record_text}
            """

        # If streaming is enabled, process the record in units
        if streaming:
            response_bool, extracted_symbols = streaming_unit_processing(
                record_text,
                regex,
                model_client,
                system_prompt,
                order_sensitive,
                verbose
            )
            pred.append(response_bool)
            true.append(label)
        else:
            # Batch processing
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

    print("\nDebuggggg")

    print("\nFinal Predictions (pred):", pred)
    print("\nGround Truth Labels (true):", true)

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

def approach2_annotate(records, model_client, dataset_name="not defined", log_results=False, annotation_prompt_template=None, extraction_prompt_template=None, system_prompt=None, manual_extraction=True, verbose=False, order_sensitive=False, streaming=False):
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
        
        # If streaming is enabled, process the record in units
        if streaming:
            # STREAMING UNIT PROCESSING
            extracted_symbols = []
            units = split_record_into_units(record_text)

            for unit_number, unit_text in enumerate(units, start=1):
                if verbose:
                    print(f"\nStreaming Unit {unit_number}:")
                    print(unit_text)
                    print("-" * 80)

                # Annotate this unit
                unit_annotation_prompt = annotation_prompt.format(regex=regex, record_text=unit_text)
                annotated_unit = model_client.generate(unit_annotation_prompt, system_prompt=system_prompt, max_tokens=1000)

                # Extract symbols from annotated unit
                if manual_extraction:
                    unit_extracted_symbols = extract_symbols_from_annotated_record(annotated_unit)
                else:
                    if extraction_prompt_template:
                        extraction_prompt = extraction_prompt_template
                    else:
                        extraction_prompt = """
                        Extract all text enclosed in semantic tags from the annotated medical record below. 

                        Return only the extracted content as a valid Python list of tuples in the format: [("tag", "text"), ...].

                        Tags may be nested. In such cases, include all relevant entries separately, even if they overlap.

                        Example:
                        Input: <patient>The <vaccine>flu vaccine</vaccine> was given.</patient>
                        Output: [("vaccine", "flu vaccine"), ("patient", "The flu vaccine was given.")]

                        Annotated Medical Record:
                        {annotated_record}
                        """

                    extraction_unit_prompt = extraction_prompt.format(annotated_record=annotated_unit)
                    extracted_text = model_client.generate(extraction_unit_prompt, max_tokens=450)
                    unit_extracted_symbols = extract_list_from_model_output(extracted_text)

                unit_extracted_symbols = [(symbol.lower(), explanation) for symbol, explanation in unit_extracted_symbols]

                # Accumulate
                extracted_symbols.extend(unit_extracted_symbols)

                # Check for match immediately
                response_bool = pattern_identification(extracted_symbols, regex, order_sensitive, verbose)

                if response_bool:
                    if verbose:
                        print(f"\n✅ Match found at streaming unit {unit_number}!")
                    break  # early stop after match
            pred.append(response_bool)
            true.append(label)
        else:
            # BATCH PROCESSING
            # Annotate the entire record
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

                    Example:
                    Input: <patient>The <vaccine>flu vaccine</vaccine> was given.</patient>
                    Output: [("vaccine", "flu vaccine"), ("patient", "The flu vaccine was given.")]

                    Annotated Medical Record:
                    {annotated_record}
                    """

                extraction_prompt = extraction_prompt.format(annotated_record=annotated_record)
                extracted_symbols = model_client.generate(extraction_prompt, max_tokens=450)
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

            pred.append(response_bool)
            true.append(label)

    end_time = time.time()
    print("\nDebuggggg")

    print("\nFinal Predictions (pred):", pred)
    print("\nGround Truth Labels (true):", true)
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

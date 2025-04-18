import re
import ast

def extract_list_from_model_output(model_output):
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

def extract_symbols_from_annotated_record(annotated_record):
  """
  This function takes an annotated medical record and extracts the semantic symbols and their corresponding text.
  """
  ordered_symbols = []
  pattern = r'<(\w+)>(.*?)</\1>'
    
  for match in re.finditer(pattern, annotated_record, re.DOTALL):
      tag = match.group(1)  # The tag name
      content = match.group(2)  # The content inside the tag
      ordered_symbols.append((tag, content))
      
  return ordered_symbols
def accuracy(pred, true):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: List of boolean predictions
        true: List of boolean ground truth values
        
    Returns:
        float: Accuracy score
    """
    if len(pred) != len(true):
        raise ValueError("Prediction and ground truth lists must have the same length")
    
    if len(pred) == 0:
        return 0.0
    
    correct = sum(p == t for p, t in zip(pred, true))
    return correct / len(pred)

def precision(pred, true):
    """
    Calculate precision: TP / (TP + FP)
    
    Args:
        pred: List of boolean predictions
        true: List of boolean ground truth values
        
    Returns:
        float: Precision score
    """
    if len(pred) != len(true):
        raise ValueError("Prediction and ground truth lists must have the same length")
    
    true_positives = sum(p and t for p, t in zip(pred, true))
    predicted_positives = sum(pred)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall(pred, true):
    """
    Calculate recall: TP / (TP + FN)
    
    Args:
        pred: List of boolean predictions
        true: List of boolean ground truth values
        
    Returns:
        float: Recall score
    """
    if len(pred) != len(true):
        raise ValueError("Prediction and ground truth lists must have the same length")
    
    true_positives = sum(p and t for p, t in zip(pred, true))
    actual_positives = sum(true)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score(pred, true):
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    
    Args:
        pred: List of boolean predictions
        true: List of boolean ground truth values
        
    Returns:
        float: F1 score
    """
    prec = precision(pred, true)
    rec = recall(pred, true)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec) 
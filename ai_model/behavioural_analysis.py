# behavioral_analysis.py
def detect_suspicious_behavior(mouse_movements, clicks, keystrokes):
    # Example: Detect rapid mouse movements
    if len(mouse_movements) > 100:  # Threshold for rapid movement
        return True
    
    # Example: Detect excessive clicks
    if len(clicks) > 50:  # Threshold for excessive clicks
        return True
    
    # Example: Detect rapid keystrokes
    if len(keystrokes) > 100:  # Threshold for rapid keystrokes
        return True
    
    return False
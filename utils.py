import torch as th
from sympy import sympify, SympifyError
import re
from typing import List

class utils:

    def __init__(self):
        pass


def get_device():
    if th.cuda.is_available():
        device = th.device("cuda")
        print("GPU is available and being used.")
    elif th.backends.mps.is_available():
        device = th.device("mps")
        print("MPS is available and being used.")
    else:
        device = th.device("cpu")
        print("GPU and MPS not available, using CPU.")
    return device


def contains_angle_brackets(text: str) -> bool:
    """
    Check if the text contains both '<<' and '>>' markers.
    
    Args:
        text: Input string to check.
    
    Returns:
        True if both '<<' and '>>' are present, False otherwise.
    """
    return ('<<' in text) and ('>>' in text)


def validate_equation(text: str, symbolic_only: bool = False) -> bool:
    """
    Hybrid validator: Checks syntax first, then uses sympy if needed.
    Args:
        text: Input text.
        symbolic_only: If True, skip numeric checks (safer).
    """
    # Quick syntax check
    if not re.search(r'^\s*([^=]+=[^=]+)\s*$', text):
        return False
    
    # Try symbolic parsing (safe)
    try:
        left, right = text.split('=', 1)
        sympify(f"Eq({left}, {right})")
        return True
    except (SympifyError, ValueError):
        return False
    

def extract_equations_from_text(text: str) -> List[str]:
    """
    Extracts mathematical equations embedded in natural language text.
    Handles cases where equations appear within sentences and dollar amounts.
    
    Args:
        text: Input text containing potential equations
        
    Returns:
        List of extracted equations with standardized formatting
    """
    # Preprocessing
    text = text.replace('$', '')  # Remove dollar signs for cleaner parsing
    
    # Enhanced pattern to catch:
    # 1. Traditional equations (3+4=7)
    # 2. Embedded calculations (spends 200-100)
    # 3. Equation-like fragments (=50+200-100)
    equation_pattern = r'''
        (?:^|\s|\.)             # Start after whitespace or period
        (                       # Capture group
            (?:                 # Left side
                [\d\.\+\-\*/\%\^\(\)\s]+
            )
            (?:                 # Optional equals/operator
                [=+\-*/^]
                [\d\.\+\-\*/\%\^\(\)\s]*
            )+
        )
        (?=\s|$|\.|,)           # Lookahead for end markers
    '''
    
    equations = []
    for match in re.finditer(equation_pattern, text, re.VERBOSE):
        equation = match.group(1).strip()
        
        # Standardize formatting
        equation = re.sub(r'\s+', '', equation)  # Remove all whitespace
        equation = equation.replace('^', '**')    # Convert ^ to **
        
        # Validate equation structure
        try:
            # Check if it's solvable (has numbers and operators)
            if not re.search(r'\d+[\+\-\*/]', equation):
                continue
                
            # Try parsing with sympy
            if '=' in equation:
                left, right = equation.split('=', 1)
                sympify(left)
                sympify(right)
            else:
                sympify(equation)
                
            equations.append(equation)
            
        except (SympifyError, ValueError, IndexError, TypeError):
            continue
    
    return equations
    

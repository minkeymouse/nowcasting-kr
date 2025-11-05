"""Utility functions for database operations."""

import numpy as np
from typing import Dict, Any, List


def serialize_numpy_array(arr: np.ndarray) -> List:
    """Serialize numpy array to list for JSON storage."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (list, tuple)):
        return [serialize_numpy_array(x) if isinstance(x, np.ndarray) else x for x in arr]
    return arr


def deserialize_numpy_array(data: List) -> np.ndarray:
    """Deserialize list to numpy array."""
    return np.array(data)


def serialize_dfm_result(result) -> Dict[str, Any]:
    """Serialize DFMResult to dictionary for database storage."""
    return {
        'parameters_json': {
            'C': serialize_numpy_array(result.C),
            'A': serialize_numpy_array(result.A),
            'Q': serialize_numpy_array(result.Q),
            'R': serialize_numpy_array(result.R),
        },
        'factors_json': {
            'Z': serialize_numpy_array(result.Z),
        },
        'standardization_json': {
            'Mx': serialize_numpy_array(result.Mx),
            'Wx': serialize_numpy_array(result.Wx),
        },
        'initial_conditions_json': {
            'Z_0': serialize_numpy_array(result.Z_0),
            'V_0': serialize_numpy_array(result.V_0),
        },
        'structure_json': {
            'r': serialize_numpy_array(result.r),
            'p': result.p,
        },
    }


def deserialize_dfm_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize database record to DFMResult-compatible dictionary."""
    result = {}
    
    if 'parameters_json' in data:
        params = data['parameters_json']
        result['C'] = deserialize_numpy_array(params.get('C', []))
        result['A'] = deserialize_numpy_array(params.get('A', []))
        result['Q'] = deserialize_numpy_array(params.get('Q', []))
        result['R'] = deserialize_numpy_array(params.get('R', []))
    
    if 'factors_json' in data:
        factors = data['factors_json']
        result['Z'] = deserialize_numpy_array(factors.get('Z', []))
    
    if 'standardization_json' in data:
        std = data['standardization_json']
        result['Mx'] = deserialize_numpy_array(std.get('Mx', []))
        result['Wx'] = deserialize_numpy_array(std.get('Wx', []))
    
    if 'initial_conditions_json' in data:
        init = data['initial_conditions_json']
        result['Z_0'] = deserialize_numpy_array(init.get('Z_0', []))
        result['V_0'] = deserialize_numpy_array(init.get('V_0', []))
    
    if 'structure_json' in data:
        struct = data['structure_json']
        result['r'] = deserialize_numpy_array(struct.get('r', []))
        result['p'] = struct.get('p', 1)
    
    return result


def map_frequency_to_code(frequency: str) -> str:
    """
    Map frequency code to internal DFM frequency code.
    
    Parameters
    ----------
    frequency : str
        Source frequency code (e.g., 'A', 'Q', 'M', 'D')
        
    Returns
    -------
    str
        Internal frequency code (d, w, m, q, sa, a)
    """
    mapping = {
        'D': 'd',   # Daily
        'SM': 'w',  # Semi-monthly → weekly
        'M': 'm',   # Monthly
        'Q': 'q',   # Quarterly
        'S': 'sa',  # Semi-annual
        'A': 'a'    # Annual
    }
    return mapping.get(frequency.upper(), frequency.lower())


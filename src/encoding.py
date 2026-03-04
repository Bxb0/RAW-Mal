"""
Payload encoding module.

This module provides functions for encoding and transforming payloads
using various techniques (XOR, padding, shuffling).
"""

from typing import List, Optional
from .config import ENCODING_OPS, XOR_KEY


def encode_payload_sequence(
    payload_data: bytes, 
    encoding_sequence: List[str],
    xor_key: int = XOR_KEY
) -> bytes:
    """
    Apply a sequence of encoding operations to payload data.
    
    The operations are applied in order, allowing for layered encoding.
    
    Args:
        payload_data: Raw payload bytes to encode.
        encoding_sequence: List of encoding operation names to apply.
        xor_key: XOR key to use for XOR operations (default: 0x42).
        
    Returns:
        Encoded payload bytes.
        
    Supported operations:
        - 'xor_full': XOR all bytes
        - 'xor_75': XOR 75% of bytes (skip every 4th byte)
        - 'xor_50': XOR 50% of bytes (every other byte)
        - 'xor_25': XOR 25% of bytes (every 4th byte)
        - 'pad_front': Add null padding at front (50% of data size)
        - 'pad_back': Add space padding at back (50% of data size)
        - 'shuffle': Interleave first and second half of data
    """
    data = bytearray(payload_data)
    
    # If no encoding operations, return original data
    if not encoding_sequence:
        return bytes(data)
    
    # Apply each encoding operation in sequence
    for op in encoding_sequence:
        if op == 'xor_full':
            for i in range(len(data)):
                data[i] ^= xor_key
        
        elif op == 'xor_50':
            for i in range(0, len(data), 2):
                data[i] ^= xor_key
        
        elif op == 'xor_25':
            for i in range(0, len(data), 4):
                data[i] ^= xor_key
        
        elif op == 'xor_75':
            for i in range(len(data)):
                if i % 4 != 3:
                    data[i] ^= xor_key
        
        elif op == 'pad_front':
            padding_size = len(data) // 2
            data = bytearray(b'\x00' * padding_size) + data
        
        elif op == 'pad_back':
            padding_size = len(data) // 2
            data = data + bytearray(b'\x20' * padding_size)
        
        elif op == 'shuffle':
            shuffled = bytearray(len(data))
            half = len(data) // 2
            for i in range(half):
                shuffled[i * 2] = data[i]
                shuffled[i * 2 + 1] = data[half + i]
            if len(data) % 2:
                shuffled[-1] = data[-1]
            data = shuffled
    
    return bytes(data)


def build_payload_header(
    encoding_sequence: List[str],
    xor_key: int = XOR_KEY
) -> bytes:
    """
    Build the header that describes the encoding operations.
    
    The header format is:
        [num_ops: 1 byte] + [op_ids: num_ops bytes] + [xor_key: 1 byte]
    
    Args:
        encoding_sequence: List of encoding operation names.
        xor_key: XOR key used for encoding.
        
    Returns:
        Header bytes.
    """
    op_ids = []
    for op in encoding_sequence:
        if op in ENCODING_OPS:
            op_ids.append(ENCODING_OPS.index(op))
    
    num_ops = len(op_ids)
    header = bytes([num_ops]) + bytes(op_ids) + bytes([xor_key])
    return header


def get_encoding_op_index(op_name: str) -> Optional[int]:
    """
    Get the index of an encoding operation by name.
    
    Args:
        op_name: Name of the encoding operation.
        
    Returns:
        Index of the operation, or None if not found.
    """
    try:
        return ENCODING_OPS.index(op_name)
    except ValueError:
        return None


def get_encoding_op_name(index: int) -> Optional[str]:
    """
    Get the name of an encoding operation by index.
    
    Args:
        index: Index of the encoding operation.
        
    Returns:
        Name of the operation, or None if index is out of range.
    """
    if 0 <= index < len(ENCODING_OPS):
        return ENCODING_OPS[index]
    return None

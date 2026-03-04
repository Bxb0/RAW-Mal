"""
PE (Portable Executable) utility functions.

This module provides functions for parsing and manipulating PE files,
including detecting architecture, extracting image base, and modifying PE headers.
"""

import struct
from typing import Optional


def detect_pe_bits(pe_data: bytes) -> int:
    """
    Detect whether a PE file is 32-bit or 64-bit.
    
    Args:
        pe_data: Raw bytes of the PE file.
        
    Returns:
        32 for 32-bit PE, 64 for 64-bit PE (default if detection fails).
    """
    try:
        if len(pe_data) < 64 or pe_data[0:2] != b'MZ':
            return 64
        pe_offset = struct.unpack('<I', pe_data[0x3C:0x40])[0]
        if len(pe_data) < pe_offset + 26:
            return 64
        magic = struct.unpack('<H', pe_data[pe_offset + 24:pe_offset + 26])[0]
        return 32 if magic == 0x10b else 64
    except Exception:
        return 64


def get_pe_image_base(pe_data: bytes) -> Optional[int]:
    """
    Get the ImageBase from a PE file's optional header.
    
    Args:
        pe_data: Raw bytes of the PE file.
        
    Returns:
        The ImageBase value, or None if extraction fails.
    """
    try:
        pe_offset = struct.unpack('<I', pe_data[0x3C:0x40])[0]
        magic = struct.unpack('<H', pe_data[pe_offset + 24:pe_offset + 26])[0]
        if magic == 0x10b:  # PE32
            return struct.unpack('<I', pe_data[pe_offset + 52:pe_offset + 56])[0]
        else:  # PE32+
            return struct.unpack('<Q', pe_data[pe_offset + 48:pe_offset + 56])[0]
    except Exception:
        return None


def get_pe_subsystem(pe_data: bytes) -> int:
    """
    Get the subsystem type from a PE file.
    
    Args:
        pe_data: Raw bytes of the PE file.
        
    Returns:
        Subsystem type: 2 for GUI (windows), 3 for Console (default).
    """
    try:
        pe_offset = struct.unpack('<I', pe_data[0x3C:0x40])[0]
        # Subsystem is at offset 68 in Optional Header
        return struct.unpack('<H', pe_data[pe_offset + 24 + 68:pe_offset + 24 + 70])[0]
    except Exception:
        return 3  # Default to console


def get_pe_size_of_image(pe_data: bytes) -> Optional[int]:
    """
    Get the SizeOfImage from a PE file's optional header.
    
    Args:
        pe_data: Raw bytes of the PE file.
        
    Returns:
        The SizeOfImage value, or None if extraction fails.
    """
    try:
        pe_offset = struct.unpack('<I', pe_data[0x3C:0x40])[0]
        return struct.unpack('<I', pe_data[pe_offset + 24 + 56:pe_offset + 24 + 60])[0]
    except Exception:
        return None


def disable_aslr(pe_path: str) -> bool:
    """
    Disable ASLR (Address Space Layout Randomization) in a PE file.
    
    This function clears the DYNAMIC_BASE flag (0x0040) in the DllCharacteristics
    field of the PE optional header.
    
    Args:
        pe_path: Path to the PE file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(pe_path, 'r+b') as f:
            f.seek(0x3C)
            pe_offset = struct.unpack('<I', f.read(4))[0]
            f.seek(pe_offset + 24 + 70)  # DllCharacteristics offset
            dll_char = struct.unpack('<H', f.read(2))[0]
            f.seek(pe_offset + 24 + 70)
            f.write(struct.pack('<H', dll_char & ~0x0040))
        return True
    except Exception:
        return False


def expand_size_of_image(pe_path: str, payload_data: bytes) -> bool:
    """
    Expand the SizeOfImage and last section's VirtualSize in a PE file.
    
    This is necessary to accommodate the payload PE in memory.
    
    Args:
        pe_path: Path to the PE file (loader).
        payload_data: Raw bytes of the payload PE.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        pe_offset_payload = struct.unpack('<I', payload_data[0x3C:0x40])[0]
        payload_size = struct.unpack(
            '<I', 
            payload_data[pe_offset_payload + 24 + 56:pe_offset_payload + 24 + 60]
        )[0]
        required_size = ((payload_size + 0x10000) // 0x1000) * 0x1000
        
        with open(pe_path, 'r+b') as f:
            f.seek(0x3C)
            pe_offset = struct.unpack('<I', f.read(4))[0]
            f.seek(pe_offset + 24 + 56)
            old_size = struct.unpack('<I', f.read(4))[0]
            
            if required_size > old_size:
                # 1. Modify SizeOfImage
                f.seek(pe_offset + 24 + 56)
                f.write(struct.pack('<I', required_size))
                
                # 2. Expand last section's VirtualSize
                f.seek(pe_offset + 6)
                num_sections = struct.unpack('<H', f.read(2))[0]
                f.seek(pe_offset + 20)
                size_of_optional_header = struct.unpack('<H', f.read(2))[0]
                
                section_header_offset = pe_offset + 24 + size_of_optional_header
                last_section_offset = section_header_offset + (num_sections - 1) * 40
                
                f.seek(last_section_offset + 12)  # VirtualAddress
                last_vaddr = struct.unpack('<I', f.read(4))[0]
                
                new_last_vsize = required_size - last_vaddr
                f.seek(last_section_offset + 8)  # VirtualSize
                f.write(struct.pack('<I', new_last_vsize))
        return True
    except Exception:
        return False


def calc_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy of byte data.
    
    Args:
        data: Input byte data.
        
    Returns:
        Entropy value between 0.0 and 8.0.
    """
    import math
    
    if not data or len(data) == 0:
        return 0.0
    
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1
    
    entropy = 0.0
    data_len = len(data)
    for count in byte_counts:
        if count > 0:
            p = count / data_len
            entropy -= p * math.log2(p)
    
    return entropy


def calc_printable_ratio(data: bytes) -> float:
    """
    Calculate the ratio of printable ASCII characters in data.
    
    Args:
        data: Input byte data.
        
    Returns:
        Ratio of printable characters (0.0 to 1.0).
    """
    if not data:
        return 0.0
    
    printable_count = sum(1 for b in data if 32 <= b <= 126)
    return printable_count / len(data)

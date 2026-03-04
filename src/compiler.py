"""
Compiler module for LLVM IR compilation and PE building.

This module provides functions for compiling loader source code to LLVM IR
and linking IR to Windows PE executables.
"""

import subprocess
import struct
from pathlib import Path
from typing import Optional, List

from .config import (
    TARGET_32, TARGET_64,
    INCLUDE_DIR_32, INCLUDE_DIR_64,
    LIB_DIR_32, LIB_DIR_64,
)
from .pe_utils import disable_aslr
from .encoding import encode_payload_sequence, build_payload_header, ENCODING_OPS


def compile_stub_to_ir(
    target_bits: int,
    stub_dir: str = "stub",
    project_root: Optional[str] = None
) -> Optional[str]:
    """
    Compile stub C source to LLVM IR bitcode.
    
    Args:
        target_bits: Target architecture (32 or 64).
        stub_dir: Directory containing stub.c.
        project_root: Project root directory for include paths.
        
    Returns:
        Path to the compiled .bc file, or None if compilation fails.
    """
    stub_path = Path(stub_dir)
    stub_path.mkdir(exist_ok=True)
    
    # Source is in stub/src/
    source_file = stub_path / "src" / "stub.c"
    
    if not source_file.exists():
        print(f"Error: Source file not found: {source_file}")
        return None
    
    # Determine target and include directory
    if target_bits == 32:
        target = TARGET_32
        include_dir = INCLUDE_DIR_32
        suffix = "_32"
    else:
        target = TARGET_64
        include_dir = INCLUDE_DIR_64
        suffix = "_64"
    
    # Build output goes to stub/build/
    build_dir = stub_path / "build"
    build_dir.mkdir(exist_ok=True)
    bc_file = build_dir / f"stub{suffix}.bc"
    
    cmd = [
        "clang", "-emit-llvm", "-c", "-O0",
        "-Xclang", "-disable-O0-optnone",  # Allow passes to work
        f"--target={target}", f"-I{include_dir}",
        str(source_file), "-o", str(bc_file)
    ]
    
    print(f"  Compiling {target_bits}-bit: {source_file} -> {bc_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    Failed: {result.stderr[:200]}")
        return None
    
    print(f"    Success")
    return str(bc_file)


def apply_pass(
    input_bc: str, 
    output_bc: str, 
    pass_name: str,
    delay_loop_plugin: Optional[str] = None
) -> bool:
    """
    Apply an LLVM optimization pass to a bitcode file.
    
    Args:
        input_bc: Path to input bitcode file.
        output_bc: Path to output bitcode file.
        pass_name: Name of the pass to apply.
        delay_loop_plugin: Path to DelayLoop.so plugin (required for 'delay-loop' pass).
        
    Returns:
        True if the pass was applied successfully, False otherwise.
    """
    if pass_name == 'delay-loop':
        if not delay_loop_plugin:
            print("Error: delay-loop pass requires plugin path")
            return False
        cmd = [
            "opt",
            f"-load-pass-plugin={delay_loop_plugin}",
            "-passes=delay-loop",
            input_bc, "-o", output_bc
        ]
    else:
        cmd = ["opt", f"-passes={pass_name}", input_bc, "-o", output_bc]
    
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def compile_to_exe(
    bc_file: str,
    output_name: str,
    bits: int,
    image_base: int,
    subsystem: int = 3
) -> Optional[str]:
    """
    Compile LLVM bitcode to Windows PE executable.
    
    Args:
        bc_file: Path to input bitcode file.
        output_name: Path for output executable.
        bits: Target architecture (32 or 64).
        image_base: ImageBase address for the executable.
        subsystem: PE subsystem (2=GUI/windows, 3=console).
        
    Returns:
        Path to the compiled executable, or None if compilation fails.
    """
    if bits == 32:
        target = TARGET_32
        lib_path = LIB_DIR_32
    else:
        target = TARGET_64
        lib_path = LIB_DIR_64
    
    subsystem_name = "windows" if subsystem == 2 else "console"
    
    cmd = [
        "clang", f"--target={target}", "-fuse-ld=lld",
        "-nostdlib", f"-Wl,--subsystem,{subsystem_name}",
        f"-Wl,--image-base=0x{image_base:X}",
        "-Wl,--entry,mainCRTStartup",
        bc_file, "-o", output_name,
        f"-L{lib_path}", "-lkernel32", "-luser32",
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        disable_aslr(output_name)
        return output_name
    return None


def attach_payload(
    loader_exe: str,
    payload_data: bytes,
    encoding_sequence: List[str],
    output_name: str
) -> Optional[str]:
    """
    Attach an encoded payload to a loader executable.
    
    The final executable format is:
        [loader exe] + [header + encoded payload] + [payload size: 4 bytes]
    
    Args:
        loader_exe: Path to the loader executable.
        payload_data: Raw payload bytes to encode and attach.
        encoding_sequence: List of encoding operations to apply.
        output_name: Path for the final executable.
        
    Returns:
        Path to the final executable, or None if operation fails.
    """
    try:
        with open(loader_exe, 'rb') as f:
            loader_data = f.read()
        
        # Encode payload
        encoded_payload = encode_payload_sequence(payload_data, encoding_sequence)
        
        # Build header
        header = build_payload_header(encoding_sequence)
        
        # Combine header and encoded payload
        payload_with_header = header + encoded_payload
        
        # Append to loader
        payload_size = struct.pack('<I', len(payload_with_header))
        final_data = loader_data + payload_with_header + payload_size
        
        with open(output_name, 'wb') as f:
            f.write(final_data)
        
        return output_name
    except Exception as e:
        print(f"Error attaching payload: {e}")
        return None

"""
AV Scanner module for interacting with antivirus detection services.

This module provides classes and functions for analyzing samples using
remote AV model and engine APIs.
"""

import json
import requests
from typing import Dict, Any, Optional

from .config import ENGINE_ADDRESS, MODEL_ADDRESS


class OnlineAntivirus:
    """Base class for online antivirus scanners."""
    
    def analyse_sample(self, sample: bytes) -> Dict[str, Any]:
        """
        Analyze a sample and return detection results.
        
        Args:
            sample: Raw bytes of the sample to analyze.
            
        Returns:
            Dictionary with 'result' (0 or 1) and 'score' (float).
        """
        raise NotImplementedError


class LocalOnlineAntivirus(OnlineAntivirus):
    """
    Antivirus scanner that communicates with a local API server.
    
    This class is used for both ML models and traditional AV engines,
    which expose their detection capabilities through HTTP APIs.
    """
    
    def __init__(self, api_url: str, engine_name: Optional[str] = None):
        """
        Initialize the scanner.
        
        Args:
            api_url: URL of the detection API endpoint.
            engine_name: Name of the engine (for multi-engine services).
        """
        self.url = api_url
        self.engine_name = engine_name

    def check_online(self, timeout: int = 5) -> None:
        """
        Check if the AV service is online.
        
        Args:
            timeout: Connection timeout in seconds.
        
        Raises:
            RuntimeError: If the service is not reachable.
        """
        try:
            # Test connection to service root
            root_url = self.url.replace('/upload_sync', '/')
            requests.get(root_url, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"AV service offline or unreachable: {e}")

    def analyse_sample(self, sample: bytes) -> Dict[str, Any]:
        """
        Analyze a sample and return detection results.
        
        Args:
            sample: Raw bytes of the sample to analyze.
            
        Returns:
            Dictionary with:
                - 'result': 0 (benign) or 1 (malicious)
                           
        Raises:
            RuntimeError: If the analysis fails.
        """
        files = {"files": sample}
        headers = {"accept": "application/json"}
        data = {}

        if self.engine_name:
            data["engine_name"] = self.engine_name

        try:
            resp = requests.post(
                self.url, 
                files=files, 
                data=data, 
                headers=headers, 
                timeout=1000
            )
            resp.raise_for_status()
            parsed = json.loads(resp.text)
        except Exception as e:
            raise RuntimeError(f"Sample analysis failed: {e}")

        result = int(parsed.get("result", -1))
        
        return {'result': result}


def build_scanner(target_type: str, target_name: str) -> LocalOnlineAntivirus:
    """
    Build a scanner for the specified target.
    
    Args:
        target_type: Type of target ('engine' or 'model').
        target_name: Name of the target (e.g., 'avastnet', 'Kaspersky').
        
    Returns:
        Configured LocalOnlineAntivirus instance.
        
    Raises:
        ValueError: If target_type or target_name is invalid.
    """
    if target_type == "engine":
        url = ENGINE_ADDRESS.get(target_name)
        if not url:
            available = ', '.join(ENGINE_ADDRESS.keys())
            raise ValueError(f"Unknown engine: {target_name}. Available: {available}")
        return LocalOnlineAntivirus(api_url=url, engine_name=target_name.lower())
    
    elif target_type == "model":
        url = MODEL_ADDRESS.get(target_name)
        if not url:
            available = ', '.join(MODEL_ADDRESS.keys())
            raise ValueError(f"Unknown model: {target_name}. Available: {available}")
        return LocalOnlineAntivirus(api_url=url)
    
    else:
        raise ValueError("target_type must be 'engine' or 'model'")


def list_available_targets() -> Dict[str, list]:
    """
    List all available scanner targets.
    
    Returns:
        Dictionary with 'engines' and 'models' lists.
    """
    return {
        'engines': list(ENGINE_ADDRESS.keys()),
        'models': list(MODEL_ADDRESS.keys())
    }

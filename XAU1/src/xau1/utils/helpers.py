"""
General utility functions
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup standardized logging configuration"""
    import sys
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate configuration has required keys"""
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    return True
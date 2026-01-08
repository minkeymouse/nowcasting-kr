"""Simple utility functions."""

import logging
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory (parent of src/)."""
    return Path(__file__).parent.parent


def setup_logging(log_dir: Path = None, force: bool = False, log_file: Path = None) -> None:
    """Setup basic logging configuration.
    
    Parameters
    ----------
    log_dir : Path, optional
        Directory for log files. If None, uses project_root/log
    force : bool, default False
        Force reconfiguration if already set up
    log_file : Path, optional
        Specific log file path. If None, logs only to console
    """
    if log_dir is None:
        log_dir = get_project_root() / "log"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic logging config
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=force
    )

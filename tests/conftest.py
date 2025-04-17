import pytest
import numpy as np
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def setup_test_environment():
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    # Ensure results directory exists for tests
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Ensure logs directory exists for tests
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Return any configuration details needed by tests
    return {"seed": 42}
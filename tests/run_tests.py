#!/usr/bin/env python
"""
Test runner script for the causal-eval project.
Runs tests with coverage reporting.
"""
import os
import sys
import pytest
import coverage

def main():
    # Start coverage measurement
    cov = coverage.Coverage(
        source=["src"],
        omit=["*/__pycache__/*", "*/test_*.py"]
    )
    cov.start()
    
    # Run pytest
    args = [
        "--verbose",
        "--color=yes",
    ]
    # Add any command line arguments passed to this script
    args.extend(sys.argv[1:])
    
    # Run the tests
    result = pytest.main(args)
    
    # Stop coverage and generate reports
    cov.stop()
    cov.save()
    cov.report()
    
    # Generate HTML report
    cov.html_report(directory="coverage_html")
    print(f"HTML coverage report generated in coverage_html/index.html")
    
    return result

if __name__ == "__main__":
    sys.exit(main())
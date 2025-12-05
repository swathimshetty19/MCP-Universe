#!/bin/bash
export PYTHONPATH=.

domains=("location_navigation" "browser_automation" "financial_analysis" 
         "repository_management" "web_search" "3d_design")

for domain in "${domains[@]}"; do
    echo "Running benchmark: $domain"
    python "tests/benchmark/test_benchmark_${domain}.py"
    echo "Completed: $domain"
done
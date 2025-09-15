#!/usr/bin/env python3
"""
Main analysis script for temporal reasoning in LLMs study.
Run this to reproduce all results.
"""

import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("Temporal Reasoning in LLMs - Complete Analysis")
    print("=" * 50)
    
    # 1. Baseline assessment
    print("\n1. Running baseline temporal discounting assessment...")
    from src.comprehensive_model_assessment import ComprehensiveModelAssessment
    assessment = ComprehensiveModelAssessment()
    baseline_results = assessment.run_comprehensive_assessment()
    
    # 2. Mechanistic tests
    print("\n2. Running mechanistic exploration...")
    from src.mechanistic_exploration_complete import CompleteMechanisticExploration
    explorer = CompleteMechanisticExploration()
    mech_results = explorer.run_complete_mechanistic_tests()
    
    # 3. Validation tests
    print("\n3. Running validation tests...")
    from src.validation_tests import ValidationTests
    validator = ValidationTests()
    val_results = validator.run_all_validations()
    
    print("\nAnalysis complete! Results saved in data/results/")

if __name__ == "__main__":
    main()

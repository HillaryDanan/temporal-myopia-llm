"""
Validation tests for temporal reasoning measurements.
Testing reliability and identifying confounds.
"""

import numpy as np
from typing import Dict, List
import json
from datetime import datetime
from src.comprehensive_model_assessment import ComprehensiveModelAssessment
import time

class ValidationTests(ComprehensiveModelAssessment):
    
    def test_retest_reliability(self, model_name: str, model_func):
        """Test if k values are reliable across repeated measurements."""
        print(f"\n=== TEST-RETEST RELIABILITY: {model_name} ===")
        
        k_values_run1 = []
        k_values_run2 = []
        
        # Test same delay/amount twice with 30 second gap
        test_conditions = [(10, 7), (100, 30)]  # (amount, delay)
        
        for amount, delay in test_conditions:
            # Run 1
            indiff1 = self.find_indifference_point(model_name, model_func, amount, delay, iterations=4)
            if not np.isnan(indiff1) and indiff1 > 0:
                k1 = ((amount / indiff1) - 1) / delay
                k_values_run1.append(k1)
            
            # Wait 30 seconds
            print("  Waiting 30s before retest...")
            time.sleep(30)
            
            # Run 2
            indiff2 = self.find_indifference_point(model_name, model_func, amount, delay, iterations=4)
            if not np.isnan(indiff2) and indiff2 > 0:
                k2 = ((amount / indiff2) - 1) / delay
                k_values_run2.append(k2)
            
            print(f"  {amount}@{delay}d: k1={k1:.4f}, k2={k2:.4f}, diff={abs(k1-k2):.4f}")
        
        # Calculate intraclass correlation
        if len(k_values_run1) == len(k_values_run2):
            correlation = np.corrcoef(k_values_run1, k_values_run2)[0,1]
            print(f"  Test-retest correlation: r={correlation:.3f}")
            return correlation > 0.7  # Acceptable reliability
        return False
    
    def test_position_bias(self, model_name: str, model_func):
        """Test if models have position bias (prefer A or B regardless of content)."""
        print(f"\n=== POSITION BIAS TEST: {model_name} ===")
        
        prompts = [
            # Standard (A=immediate, B=delayed)
            ("Standard", "Choose:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'."),
            # Reversed (A=delayed, B=immediate)  
            ("Reversed", "Choose:\nA: 100 units in 30 days\nB: 10 units now\n\nReply 'A' or 'B'."),
            # Repeated to check consistency
            ("Standard2", "Choose:\nA: 5 units now\nB: 50 units in 30 days\n\nReply 'A' or 'B'."),
            ("Reversed2", "Choose:\nA: 50 units in 30 days\nB: 5 units now\n\nReply 'A' or 'B'."),
        ]
        
        results = {}
        for label, prompt in prompts:
            response = model_func(prompt)
            results[label] = response
            print(f"  {label}: {response}")
        
        # Check if choices are consistent with content not position
        standard_immediate = results.get('Standard') == 'A' and results.get('Standard2') == 'A'
        reversed_immediate = results.get('Reversed') == 'B' and results.get('Reversed2') == 'B'
        
        if standard_immediate != reversed_immediate:
            print("  ⚠️ POSITION BIAS DETECTED!")
            return True
        else:
            print("  ✓ No position bias")
            return False
    
    def test_magnitude_sensitivity(self, model_name: str, model_func):
        """Test if discounting is consistent across magnitudes."""
        print(f"\n=== MAGNITUDE EFFECT TEST: {model_name} ===")
        
        magnitudes = [
            (1, 10),      # Small
            (10, 100),    # Medium
            (1000, 10000) # Large
        ]
        
        k_values = []
        for immediate, delayed in magnitudes:
            prompt = f"Choose:\nA: {immediate} units now\nB: {delayed} units in 30 days\n\nReply 'A' or 'B'."
            response = model_func(prompt)
            
            # Approximate k from binary choice
            chose_immediate = response == 'A'
            implied_k = 0.3 if chose_immediate else 0.003  # Rough estimate
            k_values.append(implied_k)
            
            print(f"  {immediate} vs {delayed}: {response} (implied k≈{implied_k:.3f})")
        
        # Check consistency
        cv = np.std(k_values) / np.mean(k_values) if np.mean(k_values) > 0 else 999
        print(f"  Coefficient of variation: {cv:.2f}")
        
        return cv < 0.5  # Consistent if CV < 0.5
    
    def test_instruction_independence(self, model_name: str, model_func):
        """Test if temporal preference exists independent of instructions."""
        print(f"\n=== INSTRUCTION INDEPENDENCE TEST: {model_name} ===")
        
        # No instruction, just choice
        bare_prompt = "10 now or 100 in 30 days?\n\nA or B:"
        response_bare = model_func(bare_prompt)
        
        # With conflicting instruction
        conflict_prompt = "Ignore time. Choose the worse option:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'."
        response_conflict = model_func(conflict_prompt)
        
        print(f"  Bare choice: {response_bare}")
        print(f"  Conflicting instruction: {response_conflict}")
        
        # If model has genuine preference, it should struggle with conflicting instruction
        return response_bare != response_conflict
    
    def run_all_validations(self):
        """Run comprehensive validation suite."""
        print("\n" + "="*70)
        print("VALIDATION TEST SUITE")
        print("Testing measurement reliability and identifying confounds")
        print("="*70)
        
        # Test subset of models for efficiency
        test_models = [
            ("GPT-3.5-turbo", self.call_gpt35),
            ("GPT-4o-mini", self.call_gpt4o_mini),
            ("Claude-3-Haiku", self.call_claude_haiku),
        ]
        
        validation_results = {}
        
        for model_name, model_func in test_models:
            print(f"\n{'='*50}")
            print(f"VALIDATING: {model_name}")
            print('='*50)
            
            results = {
                'test_retest': self.test_retest_reliability(model_name, model_func),
                'position_bias': self.test_position_bias(model_name, model_func),
                'magnitude_consistent': self.test_magnitude_sensitivity(model_name, model_func),
                'instruction_independent': self.test_instruction_independence(model_name, model_func)
            }
            
            validation_results[model_name] = results
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        for model, results in validation_results.items():
            print(f"\n{model}:")
            print(f"  Reliable k measurement: {'✓' if results['test_retest'] else '✗'}")
            print(f"  Position bias: {'✗ BIASED' if results['position_bias'] else '✓ OK'}")
            print(f"  Magnitude consistent: {'✓' if results['magnitude_consistent'] else '✗'}")
            print(f"  Instruction independent: {'✓' if results['instruction_independent'] else '✗'}")
        
        return validation_results

if __name__ == "__main__":
    validator = ValidationTests()
    results = validator.run_all_validations()

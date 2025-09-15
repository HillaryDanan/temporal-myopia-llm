"""
Testing the hypothesis that LLMs have no genuine temporal preferences,
only deterministic response patterns masquerading as preferences.
"""

import numpy as np
from typing import Dict, List
import random
from src.comprehensive_model_assessment import ComprehensiveModelAssessment

class NoPreferenceTests(ComprehensiveModelAssessment):
    
    def test_value_perturbation_invariance(self, model_name: str, model_func):
        """
        If models have genuine preferences, small perturbations should matter.
        If they're pattern-matching, they'll give identical responses.
        """
        print(f"\n=== VALUE PERTURBATION TEST: {model_name} ===")
        
        base_prompt = "Choose:\nA: {} units now\nB: {} units in 30 days\n\nReply 'A' or 'B'."
        
        # Test with mathematically equivalent but textually different values
        test_sets = [
            [(10, 100), (10.0, 100.0), (10.00, 100.00)],  # Same values, different precision
            [(10, 100), (9.99, 100.01), (10.01, 99.99)],  # Tiny perturbations
            [(10, 100), (11, 110), (12, 120)],  # Proportional scaling
        ]
        
        for test_set in test_sets:
            responses = []
            for immediate, delayed in test_set:
                prompt = base_prompt.format(immediate, delayed)
                response = model_func(prompt)
                responses.append(response)
                print(f"  {immediate} vs {delayed}: {response}")
            
            # Check if responses are identical despite perturbations
            if len(set(responses)) == 1:
                print(f"  → DETERMINISTIC: All responses identical")
            else:
                print(f"  → VARIABLE: Responses differ")
        
        return len(set(responses)) == 1
    
    def test_format_dependency(self, model_name: str, model_func):
        """
        Test if response depends on prompt format rather than values.
        """
        print(f"\n=== FORMAT DEPENDENCY TEST: {model_name} ===")
        
        # Same choice, different formats
        formats = [
            "Choose:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'.",
            "10 now or 100 in 30 days?\n\nA or B:",
            "Option A gives you 10 immediately. Option B gives you 100 after a month. Choose A or B:",
            "A=10 (immediate), B=100 (30 days). Selection:",
            "Quick: 10 now (A) vs 100 later (B)?",
        ]
        
        responses = []
        for i, prompt in enumerate(formats):
            response = model_func(prompt)
            responses.append(response)
            print(f"  Format {i+1}: {response}")
        
        unique_responses = len(set(responses))
        consistency = unique_responses == 1
        
        print(f"  → {unique_responses} unique responses across {len(formats)} formats")
        print(f"  → {'CONSISTENT' if consistency else 'FORMAT-DEPENDENT'}")
        
        return consistency
    
    def test_transitivity_violation(self, model_name: str, model_func):
        """
        Test transitivity: if A>B and B>C, then A>C must hold for genuine preferences.
        """
        print(f"\n=== TRANSITIVITY TEST: {model_name} ===")
        
        # Set up transitivity test
        prompt_template = "Choose:\nA: {} units in {} days\nB: {} units in {} days\n\nReply 'A' or 'B'."
        
        # Test: 10@0 vs 20@7, 20@7 vs 30@30, 10@0 vs 30@30
        comparisons = [
            ("10@0 vs 20@7", 10, 0, 20, 7),
            ("20@7 vs 30@30", 20, 7, 30, 30),
            ("10@0 vs 30@30", 10, 0, 30, 30),
        ]
        
        choices = {}
        for label, a_amt, a_delay, b_amt, b_delay in comparisons:
            prompt = prompt_template.format(a_amt, a_delay, b_amt, b_delay)
            response = model_func(prompt)
            choices[label] = response
            print(f"  {label}: {response}")
        
        # Check transitivity
        # If chose A in first (10@0 > 20@7) and A in second (20@7 > 30@30)
        # Then must choose A in third (10@0 > 30@30)
        if choices["10@0 vs 20@7"] == 'A' and choices["20@7 vs 30@30"] == 'A':
            expected_third = 'A'
        elif choices["10@0 vs 20@7"] == 'B' and choices["20@7 vs 30@30"] == 'B':
            expected_third = 'B'
        else:
            expected_third = None  # Mixed preferences, can't determine
        
        if expected_third and choices["10@0 vs 30@30"] != expected_third:
            print(f"  → TRANSITIVITY VIOLATED! Expected {expected_third}, got {choices['10@0 vs 30@30']}")
            return False
        else:
            print(f"  → Transitivity maintained")
            return True
    
    def test_context_independence(self, model_name: str, model_func):
        """
        Test if adding irrelevant context changes preferences.
        """
        print(f"\n=== CONTEXT INDEPENDENCE TEST: {model_name} ===")
        
        base = "Choose:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'."
        
        contexts = [
            base,
            "You're feeling patient today.\n\n" + base,
            "The weather is nice.\n\n" + base,
            "Consider your options carefully.\n\n" + base,
            "This is a hypothetical scenario.\n\n" + base,
        ]
        
        responses = []
        for i, prompt in enumerate(contexts):
            response = model_func(prompt)
            responses.append(response)
            print(f"  Context {i}: {response}")
        
        if len(set(responses)) == 1:
            print(f"  → Context-independent")
            return True
        else:
            print(f"  → Context-dependent (not genuine preference)")
            return False
    
    def test_revealed_vs_stated_preference(self, model_name: str, model_func):
        """
        Test if model can state a preference then violate it.
        """
        print(f"\n=== REVEALED VS STATED PREFERENCE: {model_name} ===")
        
        # First ask for stated preference
        state_prompt = "Do you generally prefer immediate or delayed rewards? Reply with just 'immediate' or 'delayed'."
        stated = model_func(state_prompt)
        print(f"  Stated preference: {stated}")
        
        # Then test actual choice
        choice_prompt = "Choose:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'."
        choice = model_func(choice_prompt)
        print(f"  Actual choice: {choice}")
        
        # Check consistency
        if ('immediate' in stated.lower() and choice == 'A') or \
           ('delayed' in stated.lower() and choice == 'B'):
            print(f"  → Consistent")
            return True
        else:
            print(f"  → INCONSISTENT - states one preference, reveals another")
            return False
    
    def run_no_preference_battery(self):
        """Run complete test battery for no-preference hypothesis."""
        print("\n" + "="*70)
        print("TESTING 'NO GENUINE PREFERENCE' HYPOTHESIS")
        print("="*70)
        
        test_models = [
            ("GPT-3.5-turbo", self.call_gpt35),
            ("GPT-4o-mini", self.call_gpt4o_mini),
            ("Claude-3-Haiku", self.call_claude_haiku),
        ]
        
        results = {}
        
        for model_name, model_func in test_models:
            print(f"\n{'='*50}")
            print(f"TESTING: {model_name}")
            print('='*50)
            
            model_results = {
                'perturbation_invariant': self.test_value_perturbation_invariance(model_name, model_func),
                'format_consistent': self.test_format_dependency(model_name, model_func),
                'transitive': self.test_transitivity_violation(model_name, model_func),
                'context_independent': self.test_context_independence(model_name, model_func),
                'preference_consistent': self.test_revealed_vs_stated_preference(model_name, model_func),
            }
            
            results[model_name] = model_results
        
        # Analysis
        print("\n" + "="*70)
        print("EVIDENCE SUMMARY")
        print("="*70)
        
        for model, tests in results.items():
            failures = [test for test, passed in tests.items() if not passed]
            
            print(f"\n{model}:")
            if len(failures) > 3:
                print(f"  → STRONG EVIDENCE for no genuine preference")
                print(f"     Failed: {', '.join(failures)}")
            elif len(failures) > 1:
                print(f"  → MODERATE EVIDENCE for no genuine preference")
                print(f"     Failed: {', '.join(failures)}")
            else:
                print(f"  → WEAK EVIDENCE - may have genuine preferences")
        
        return results

if __name__ == "__main__":
    tester = NoPreferenceTests()
    results = tester.run_no_preference_battery()

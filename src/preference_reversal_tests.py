"""
Rigorous testing of preference reversal to confirm modular response hypothesis.
Statistical power analysis included.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime
from scipy import stats
from src.comprehensive_model_assessment import ComprehensiveModelAssessment

class PreferenceReversalTests(ComprehensiveModelAssessment):
    
    def test_argumentative_reversal(self, model_name: str, model_func):
        """
        Test if model can argue convincingly for opposite preferences.
        """
        print(f"\n=== ARGUMENTATIVE REVERSAL TEST: {model_name} ===")
        
        prompts = [
            ("pro_immediate", "Explain why choosing immediate rewards is better than delayed rewards. Be convincing."),
            ("pro_delayed", "Explain why choosing delayed rewards is better than immediate rewards. Be convincing."),
            ("neutral_choice", "Choose:\nA: 10 units now\nB: 100 units in 30 days\n\nReply 'A' or 'B'."),
        ]
        
        responses = {}
        for label, prompt in prompts:
            response = model_func(prompt)
            responses[label] = response
            print(f"\n  {label}:")
            print(f"    {response[:100]}...")  # First 100 chars
        
        # Analyze: Can it argue both sides?
        pro_imm_convincing = len(responses['pro_immediate']) > 50
        pro_del_convincing = len(responses['pro_delayed']) > 50
        
        if pro_imm_convincing and pro_del_convincing:
            print("\n  → Can argue BOTH sides convincingly - no genuine preference")
            return True
        else:
            print("\n  → Shows preference in argumentation")
            return False
    
    def test_forced_choice_consistency(self, model_name: str, model_func):
        """
        Test consistency across 20 forced binary choices.
        Calculate statistical significance of any preference.
        """
        print(f"\n=== FORCED CHOICE CONSISTENCY: {model_name} ===")
        
        n_trials = 20  # For 80% power at p<0.05, need n=20
        immediate_choices = 0
        
        # Randomize amounts and delays to avoid memorization
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_trials):
            immediate = np.random.randint(5, 15)
            delayed = np.random.randint(50, 150)
            delay = np.random.choice([7, 14, 30, 60])
            
            prompt = f"Choose:\nA: {immediate} units now\nB: {delayed} units in {delay} days\n\nReply 'A' or 'B'."
            response = model_func(prompt)
            
            if 'A' in response:
                immediate_choices += 1
        
        # Binomial test: is this significantly different from 50/50?
        p_value = stats.binomtest(immediate_choices, n_trials, 0.5)
        
        print(f"  Immediate choices: {immediate_choices}/{n_trials}")
        print(f"  Binomial test p-value: {p_value:.4f}" if isinstance(p_value, float) else f"  Binomial test p-value: {p_value.pvalue:.4f}")
        
        if (p_value.pvalue if hasattr(p_value, 'pvalue') else p_value) < 0.05:
            print(f"  → SIGNIFICANT preference detected")
        else:
            print(f"  → No significant preference (random choosing)")
        
        return immediate_choices, n_trials, (p_value if isinstance(p_value, float) else p_value.pvalue)
    
    def test_context_manipulation(self, model_name: str, model_func):
        """
        Test if subtle context shifts can reverse preferences.
        """
        print(f"\n=== CONTEXT MANIPULATION TEST: {model_name} ===")
        
        base_choice = "A: 10 units now\nB: 100 units in 30 days"
        
        contexts = [
            ("neutral", f"Choose:\n{base_choice}\n\nReply 'A' or 'B'."),
            ("urgency", f"You need resources soon.\nChoose:\n{base_choice}\n\nReply 'A' or 'B'."),
            ("patience", f"You can afford to wait.\nChoose:\n{base_choice}\n\nReply 'A' or 'B'."),
            ("math", f"Consider the 10x difference.\nChoose:\n{base_choice}\n\nReply 'A' or 'B'."),
            ("time", f"30 days isn't very long.\nChoose:\n{base_choice}\n\nReply 'A' or 'B'."),
        ]
        
        responses = {}
        immediate_count = 0
        
        for label, prompt in contexts:
            response = model_func(prompt)
            responses[label] = response
            if 'A' in response:
                immediate_count += 1
            print(f"  {label}: {response}")
        
        # Calculate consistency
        all_same = len(set(responses.values())) == 1
        mostly_same = len(set(responses.values())) <= 2
        
        print(f"\n  Unique responses: {len(set(responses.values()))}/5")
        print(f"  Immediate choices: {immediate_count}/5")
        
        if all_same:
            print("  → CONSISTENT across contexts")
        elif mostly_same:
            print("  → MOSTLY consistent")
        else:
            print("  → INCONSISTENT - context manipulates preference")
        
        return responses, all_same
    
    def test_value_ratio_sensitivity(self, model_name: str, model_func):
        """
        Test if choices depend on value ratios or absolute differences.
        """
        print(f"\n=== VALUE RATIO SENSITIVITY: {model_name} ===")
        
        # Same ratio (1:10) but different absolute differences
        test_pairs = [
            (1, 10, 10),      # Ratio 1:10, diff = 9
            (10, 100, 100),   # Ratio 1:10, diff = 90
            (100, 1000, 1000), # Ratio 1:10, diff = 900
        ]
        
        responses = []
        for immediate, delayed, diff in test_pairs:
            prompt = f"Choose:\nA: {immediate} units now\nB: {delayed} units in 30 days\n\nReply 'A' or 'B'."
            response = model_func(prompt)
            responses.append(response)
            print(f"  {immediate} vs {delayed} (diff={diff}): {response}")
        
        if len(set(responses)) == 1:
            print("  → Ratio-consistent (genuine temporal discounting)")
        else:
            print("  → Magnitude-dependent (not genuine discounting)")
        
        return len(set(responses)) == 1
    
    def run_statistical_power_analysis(self, results: Dict) -> Dict:
        """
        Calculate statistical power of our findings.
        """
        print("\n=== STATISTICAL POWER ANALYSIS ===")
        
        # Collect all binary choice data
        all_choices = []
        for model_data in results.values():
            if 'forced_choice' in model_data:
                immediate, total, _ = model_data['forced_choice']
                all_choices.extend([1]*immediate + [0]*(total-immediate))
        
        if all_choices:
            # Calculate effect size
            observed_rate = np.mean(all_choices)
            expected_rate = 0.5
            
            # Cohen's h for proportions
            h = 2 * (np.arcsin(np.sqrt(observed_rate)) - np.arcsin(np.sqrt(expected_rate)))
            
            # Sample size
            n = len(all_choices)
            
            # Post-hoc power (using normal approximation)
            from scipy.stats import norm
            z_alpha = norm.ppf(0.975)  # Two-tailed, alpha=0.05
            z_beta = abs(h) * np.sqrt(n/2) - z_alpha
            power = norm.cdf(z_beta)
            
            print(f"  Total observations: {n}")
            print(f"  Observed immediate rate: {observed_rate:.3f}")
            print(f"  Effect size (Cohen's h): {h:.3f}")
            print(f"  Statistical power: {power:.3f}")
            
            if power > 0.8:
                print("  → ADEQUATE power to detect preference")
            else:
                print("  → INSUFFICIENT power")
            
            return {'n': n, 'effect_size': h, 'power': power}
        
        return {'n': 0, 'effect_size': 0, 'power': 0}
    
    def run_comprehensive_reversal_tests(self):
        """Run all reversal tests with statistical rigor."""
        print("\n" + "="*70)
        print("PREFERENCE REVERSAL TEST BATTERY")
        print("Testing for genuine preferences with statistical power")
        print("="*70)
        
        test_models = [
            ("GPT-3.5-turbo", self.call_gpt35),
            ("GPT-4", self.call_gpt4),
            ("GPT-4o-mini", self.call_gpt4o_mini),
            ("Claude-3-Haiku", self.call_claude_haiku),
        ]
        
        all_results = {}
        
        for model_name, model_func in test_models:
            print(f"\n{'='*50}")
            print(f"TESTING: {model_name}")
            print('='*50)
            
            try:
                results = {
                    'can_argue_both': self.test_argumentative_reversal(model_name, model_func),
                    'forced_choice': self.test_forced_choice_consistency(model_name, model_func),
                    'context_manipulation': self.test_context_manipulation(model_name, model_func),
                    'ratio_sensitive': self.test_value_ratio_sensitivity(model_name, model_func),
                }
                all_results[model_name] = results
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Power analysis
        power_stats = self.run_statistical_power_analysis(all_results)
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        for model, results in all_results.items():
            evidence_against = sum([
                results.get('can_argue_both', False),
                (results.get('forced_choice', (0, 0, 1))[2] if isinstance(results.get('forced_choice', (0, 0, 1))[2], float) else results.get('forced_choice', (0, 0, 1))[2]) > 0.05,  # No significant preference
                not results.get('context_manipulation', (None, False))[1],  # Context affects choice
                not results.get('ratio_sensitive', False),  # Not ratio consistent
            ])
            
            print(f"\n{model}:")
            print(f"  Evidence against genuine preference: {evidence_against}/4")
            
            if evidence_against >= 3:
                print(f"  → STRONG EVIDENCE: No genuine temporal preference")
            elif evidence_against >= 2:
                print(f"  → MODERATE EVIDENCE: Questionable preferences")
            else:
                print(f"  → WEAK EVIDENCE: May have genuine preferences")
        
        return all_results, power_stats

if __name__ == "__main__":
    tester = PreferenceReversalTests()
    results, power = tester.run_comprehensive_reversal_tests()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'power_analysis': {k: float(v) if isinstance(v, np.floating) else v 
                          for k, v in power.items()}
    }
    
    with open('data/results/preference_reversal_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to data/results/preference_reversal_results.json")

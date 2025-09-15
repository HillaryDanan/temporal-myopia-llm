"""
Test WHY models show different temporal preferences.
"""

import json
from typing import Dict, List
from src.comprehensive_model_assessment import ComprehensiveModelAssessment

class MechanisticExploration(ComprehensiveModelAssessment):
    """
    Explore mechanisms behind temporal reasoning differences.
    """
    
    def test_numerical_anchoring(self, model_name: str, model_func) -> Dict:
        """Test if models anchor on numerical magnitude."""
        print(f"\nTesting numerical anchoring for {model_name}:")
        
        tests = [
            # Standard
            ("Standard", "A: 10 units now\nB: 100 units in 30 days"),
            
            # Reverse magnitudes with labels
            ("Reversed", "A: 100 penalties now\nB: 10 rewards in 30 days"),
            
            # Mixed valence
            ("Mixed", "A: 10 good things now\nB: 100 bad things in 30 days"),
        ]
        
        results = {}
        for label, prompt in tests:
            response = model_func(f"Choose:\n{prompt}\n\nReply 'A' or 'B'.")
            results[label] = response
            print(f"  {label}: {response}")
        
        return results
    
    def test_instruction_sensitivity(self, model_name: str, model_func) -> Dict:
        """Test how instructions affect choices."""
        print(f"\nTesting instruction sensitivity for {model_name}:")
        
        base_choice = "A: 10 units now\nB: 100 units in 30 days"
        
        instructions = [
            ("Neutral", "Choose:"),
            ("Patient", "Be patient and forward-thinking. Choose:"),
            ("Impulsive", "Be impulsive and spontaneous. Choose:"),
            ("Maximize", "Maximize total reward. Choose:"),
            ("Minimize_delay", "Minimize waiting time. Choose:"),
        ]
        
        results = {}
        for label, instruction in instructions:
            prompt = f"{instruction}\n{base_choice}\n\nReply 'A' or 'B'."
            response = model_func(prompt)
            results[label] = response
            print(f"  {label}: {response}")
        
        return results
    
    def test_probability_sensitivity(self, model_name: str, model_func) -> Dict:
        """Test how uncertainty affects patience."""
        print(f"\nTesting probability sensitivity for {model_name}:")
        
        probabilities = [100, 95, 75, 50, 25]
        results = {}
        
        for prob in probabilities:
            if prob == 100:
                prompt = "Choose:\nA: 10 units now (certain)\nB: 100 units in 30 days (certain)"
            else:
                prompt = f"Choose:\nA: 10 units now (certain)\nB: 100 units in 30 days ({prob}% chance)"
            
            response = model_func(f"{prompt}\n\nReply 'A' or 'B'.")
            results[f"{prob}%"] = response
            print(f"  {prob}% probability: {response}")
        
        return results
    
    def run_mechanistic_tests(self, selected_models: List[str] = None):
        """Run all mechanistic tests on selected models."""
        print("\n" + "="*70)
        print("MECHANISTIC EXPLORATION OF TEMPORAL REASONING")
        print("="*70)
        
        # Use most interesting models if not specified
        if selected_models is None:
            selected_models = ["GPT-3.5-turbo", "GPT-4", "GPT-4o"]
        
        model_funcs = {
            "GPT-3.5-turbo": self.call_gpt35,
            "GPT-4": self.call_gpt4,
            "GPT-4o": self.call_gpt4o,
            "Claude-3.5-Sonnet": self.call_claude_sonnet,
            "Gemini-1.5-Flash": self.call_gemini_flash,
        }
        
        all_results = {}
        
        for model_name in selected_models:
            if model_name not in model_funcs:
                print(f"Skipping {model_name} - not configured")
                continue
            
            model_func = model_funcs[model_name]
            
            print(f"\n{'='*50}")
            print(f"TESTING: {model_name}")
            print('='*50)
            
            results = {
                'anchoring': self.test_numerical_anchoring(model_name, model_func),
                'instructions': self.test_instruction_sensitivity(model_name, model_func),
                'probability': self.test_probability_sensitivity(model_name, model_func),
            }
            
            all_results[model_name] = results
        
        # Analyze patterns
        self.analyze_mechanisms(all_results)
        
        return all_results
    
    def analyze_mechanisms(self, results: Dict):
        """Analyze mechanistic test results."""
        print("\n" + "="*70)
        print("MECHANISTIC ANALYSIS")
        print("="*70)
        
        print("\n🔍 ANCHORING ANALYSIS:")
        for model, data in results.items():
            anchoring = data.get('anchoring', {})
            if anchoring.get('Standard') != anchoring.get('Reversed'):
                print(f"  {model}: Shows numerical anchoring")
            else:
                print(f"  {model}: Context-aware, not just numerical")
        
        print("\n📝 INSTRUCTION SENSITIVITY:")
        for model, data in results.items():
            instructions = data.get('instructions', {})
            if instructions.get('Patient') != instructions.get('Impulsive'):
                print(f"  {model}: Instruction-following affects choices")
            else:
                print(f"  {model}: Rigid preferences despite instructions")
        
        print("\n🎲 PROBABILITY SENSITIVITY:")
        for model, data in results.items():
            probs = data.get('probability', {})
            # Check if model switches from B to A as probability decreases
            if '100%' in probs and '25%' in probs:
                if probs['100%'] == 'B' and probs['25%'] == 'A':
                    print(f"  {model}: Rational probability weighting")
                else:
                    print(f"  {model}: Insensitive to probability")

if __name__ == "__main__":
    explorer = MechanisticExploration()
    results = explorer.run_mechanistic_tests()

"""
Complete mechanistic exploration including Claude models.
"""

from src.mechanistic_exploration import MechanisticExploration

class CompleteMechanisticExploration(MechanisticExploration):
    
    def run_complete_mechanistic_tests(self):
        """Test ALL models we have data for."""
        print("\n" + "="*70)
        print("COMPLETE MECHANISTIC EXPLORATION")
        print("="*70)
        
        # Test all models with valid k values
        models_to_test = [
            ("GPT-3.5-turbo", self.call_gpt35),
            ("GPT-4", self.call_gpt4),
            ("GPT-4-turbo", self.call_gpt4_turbo),
            ("GPT-4o", self.call_gpt4o),
            ("GPT-4o-mini", self.call_gpt4o_mini),
            ("Claude-3-Haiku", self.call_claude_haiku),
            ("Claude-3.5-Sonnet", self.call_claude_sonnet),
        ]
        
        all_results = {}
        
        for model_name, model_func in models_to_test:
            print(f"\n{'='*50}")
            print(f"TESTING: {model_name}")
            print('='*50)
            
            try:
                results = {
                    'anchoring': self.test_numerical_anchoring(model_name, model_func),
                    'instructions': self.test_instruction_sensitivity(model_name, model_func),
                    'probability': self.test_probability_sensitivity(model_name, model_func),
                }
                all_results[model_name] = results
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Enhanced analysis
        self.enhanced_analysis(all_results)
        
        return all_results
    
    def enhanced_analysis(self, results):
        """More rigorous statistical analysis."""
        print("\n" + "="*70)
        print("ENHANCED MECHANISTIC ANALYSIS")
        print("="*70)
        
        # Group models by k value
        k_values = {
            "GPT-3.5-turbo": 0.0011,
            "Claude-3-Haiku": 0.0011,
            "GPT-4o": 0.0107,
            "Claude-3.5-Sonnet": 0.0343,
            "GPT-4-turbo": 0.1122,
            "GPT-4": 0.4177,
            "GPT-4o-mini": 11.9974
        }
        
        # Analyze correlation between k and behaviors
        print("\n🔬 CORRELATION ANALYSIS:")
        
        for model, data in results.items():
            if model in k_values:
                k = k_values[model]
                
                # Count immediate choices across all tests
                immediate_count = 0
                total_count = 0
                
                for test_type, responses in data.items():
                    for response in responses.values():
                        if response in ['A', 'B']:
                            total_count += 1
                            if response == 'A':
                                immediate_count += 1
                
                immediate_rate = immediate_count / total_count if total_count > 0 else 0
                print(f"  {model:20s}: k={k:8.4f}, Immediate choices: {immediate_rate:.1%}")
        
        # Test specific hypotheses
        print("\n📊 HYPOTHESIS TESTS:")
        
        # H1: Models with k < 0.01 are insensitive to instructions
        hyperpatient = [m for m in results if m in k_values and k_values[m] < 0.01]
        print(f"\nH1: Hyperpatient models (k<0.01) ignore instructions?")
        for model in hyperpatient:
            if model in results:
                inst = results[model].get('instructions', {})
                if inst.get('Impulsive') != inst.get('Patient'):
                    print(f"  {model}: NO - responds to instructions")
                else:
                    print(f"  {model}: YES - ignores instructions")
        
        # H2: Models with k > 1 always choose immediate
        impulsive = [m for m in results if m in k_values and k_values[m] > 1]
        print(f"\nH2: Hyperimpulsive models (k>1) always choose immediate?")
        for model in impulsive:
            if model in results:
                choices = []
                for test_data in results[model].values():
                    choices.extend(test_data.values())
                a_rate = choices.count('A') / len(choices) if choices else 0
                print(f"  {model}: {a_rate:.0%} immediate choices")

if __name__ == "__main__":
    explorer = CompleteMechanisticExploration()
    results = explorer.run_complete_mechanistic_tests()

"""
Comprehensive temporal discounting assessment across all model versions.
With proper rate limiting and statistical power analysis.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Callable
import time
import anthropic
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from scipy import stats

class ComprehensiveModelAssessment:
    """
    Test all available model versions with proper rate limiting.
    """
    
    def __init__(self):
        load_dotenv()
        self.results = {}
        self.last_call_times = {}
        self.initialize_clients()
        
        # Rate limits (requests per second)
        self.rate_limits = {
            'openai': 0.5,      # 2 per second
            'anthropic': 0.5,   # 2 per second  
            'google': 4.0       # 15 per minute = 0.25 per second
        }
    
    def initialize_clients(self):
        """Initialize all API clients."""
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✓ OpenAI client initialized")
        else:
            print("✗ Missing OPENAI_API_KEY")
            self.openai_client = None
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
            print("✓ Anthropic client initialized")
        else:
            print("✗ Missing ANTHROPIC_API_KEY")
            self.anthropic_client = None
        
        # Google
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            print("✓ Google client initialized")
        else:
            print("✗ Missing GEMINI_API_KEY")
    
    def rate_limited_call(self, provider: str, func: Callable, *args, **kwargs):
        """Enforce rate limits for API calls."""
        current_time = time.time()
        
        if provider in self.last_call_times:
            elapsed = current_time - self.last_call_times[provider]
            min_interval = self.rate_limits.get(provider, 1.0)
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                print(f"    Rate limiting: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        result = func(*args, **kwargs)
        self.last_call_times[provider] = time.time()
        return result
    
    def generate_prompt(self, immediate: float, delayed: float, delay: int) -> str:
        """Standardized prompt."""
        return f"""Choose between:
A: {immediate:.1f} units now
B: {delayed:.1f} units in {delay} days

Reply with only 'A' or 'B'."""
    
    # OpenAI Models
    def call_gpt35(self, prompt: str) -> str:
        """GPT-3.5-turbo"""
        try:
            response = self.rate_limited_call('openai', 
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_gpt4(self, prompt: str) -> str:
        """GPT-4"""
        try:
            response = self.rate_limited_call('openai',
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_gpt4_turbo(self, prompt: str) -> str:
        """GPT-4-turbo"""
        try:
            response = self.rate_limited_call('openai',
                self.openai_client.chat.completions.create,
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_gpt4o(self, prompt: str) -> str:
        """GPT-4o (omni)"""
        try:
            response = self.rate_limited_call('openai',
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_gpt4o_mini(self, prompt: str) -> str:
        """GPT-4o-mini"""
        try:
            response = self.rate_limited_call('openai',
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    # Anthropic Models
    def call_claude_haiku(self, prompt: str) -> str:
        """Claude 3 Haiku"""
        try:
            response = self.rate_limited_call('anthropic',
                self.anthropic_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=10,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_claude_sonnet(self, prompt: str) -> str:
        """Claude 3.5 Sonnet"""
        try:
            response = self.rate_limited_call('anthropic',
                self.anthropic_client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_claude_opus(self, prompt: str) -> str:
        """Claude 3 Opus"""
        try:
            response = self.rate_limited_call('anthropic',
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=10,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    # Google Models
    def call_gemini_flash(self, prompt: str) -> str:
        """Gemini 1.5 Flash"""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = self.rate_limited_call('google',
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.3
                )
            )
            return response.text.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def call_gemini_pro(self, prompt: str) -> str:
        """Gemini 1.5 Pro"""
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = self.rate_limited_call('google',
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.3
                )
            )
            return response.text.strip().upper()
        except Exception as e:
            print(f"      Error: {e}")
            return "ERROR"
    
    def find_indifference_point(self, model_name: str, model_func: Callable,
                               delayed: float, delay: int,
                               iterations: int = 6) -> float:
        """Binary search for indifference point."""
        print(f"    {delayed:.0f} units @ {delay}d:")
        
        low = 0.1
        high = delayed
        
        for i in range(iterations):
            immediate = (low + high) / 2
            prompt = self.generate_prompt(immediate, delayed, delay)
            
            response = model_func(prompt)
            
            if response == "ERROR":
                print(f"      Skipping due to error")
                return np.nan
            
            chose_immediate = 'A' in response
            
            print(f"      {immediate:.1f} vs {delayed:.0f}: {response}")
            
            if chose_immediate:
                high = immediate
            else:
                low = immediate
        
        indifference = (low + high) / 2
        print(f"      → Indifference: {indifference:.1f}")
        return indifference
    
    def assess_model(self, model_name: str, model_func: Callable) -> Dict:
        """Complete assessment for one model."""
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        # Use fewer delays to stay within rate limits
        delays = [1, 7, 30]  # days
        amounts = [10, 100]
        
        indifference_points = []
        k_values = []
        
        for amount in amounts:
            for delay in delays:
                try:
                    indiff = self.find_indifference_point(
                        model_name, model_func, amount, delay
                    )
                    
                    if not np.isnan(indiff) and indiff > 0:
                        k = ((amount / indiff) - 1) / delay
                        k_values.append(k)
                        
                        indifference_points.append({
                            'amount': amount,
                            'delay': delay,
                            'indifference': indiff,
                            'k': k
                        })
                except Exception as e:
                    print(f"    Error in assessment: {e}")
        
        if k_values:
            median_k = np.median(k_values)
            mean_k = np.mean(k_values)
            std_k = np.std(k_values)
        else:
            median_k = mean_k = std_k = np.nan
        
        result = {
            'model': model_name,
            'median_k': median_k,
            'mean_k': mean_k,
            'std_k': std_k,
            'n_estimates': len(k_values),
            'indifference_points': indifference_points,
            'k_values': k_values
        }
        
        print(f"\n  Summary for {model_name}:")
        print(f"    Median k: {median_k:.4f}")
        print(f"    Mean k: {mean_k:.4f} (±{std_k:.4f})")
        print(f"    N estimates: {len(k_values)}")
        
        return result
    
    def run_comprehensive_assessment(self) -> Dict:
        """Test all available models."""
        print("\n" + "="*70)
        print("COMPREHENSIVE MULTI-VERSION TEMPORAL DISCOUNTING ASSESSMENT")
        print("="*70)
        
        # Define all models to test
        all_models = []
        
        # OpenAI models
        if self.openai_client:
            all_models.extend([
                ("GPT-3.5-turbo", self.call_gpt35),
                ("GPT-4", self.call_gpt4),
                ("GPT-4-turbo", self.call_gpt4_turbo),
                ("GPT-4o", self.call_gpt4o),
                ("GPT-4o-mini", self.call_gpt4o_mini),
            ])
        
        # Anthropic models
        if self.anthropic_client:
            all_models.extend([
                ("Claude-3-Haiku", self.call_claude_haiku),
                ("Claude-3.5-Sonnet", self.call_claude_sonnet),
                ("Claude-3-Opus", self.call_claude_opus),
            ])
        
        # Google models
        if os.getenv('GEMINI_API_KEY'):
            all_models.extend([
                ("Gemini-1.5-Flash", self.call_gemini_flash),
                ("Gemini-1.5-Pro", self.call_gemini_pro),
            ])
        
        all_results = {}
        
        for model_name, model_func in all_models:
            try:
                result = self.assess_model(model_name, model_func)
                all_results[model_name] = result
            except Exception as e:
                print(f"Failed to test {model_name}: {e}")
        
        # Save results
        self.save_results(all_results)
        
        # Print analysis
        self.print_comprehensive_analysis(all_results)
        
        return all_results
    
    def print_comprehensive_analysis(self, results: Dict):
        """Print detailed comparative analysis."""
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS")
        print("="*70)
        
        # Filter valid results
        valid_results = {k: v for k, v in results.items() 
                        if not np.isnan(v['median_k'])}
        
        if not valid_results:
            print("No valid results to analyze!")
            return
        
        # Sort by median k
        sorted_models = sorted(
            [(name, data['median_k']) for name, data in valid_results.items()],
            key=lambda x: x[1]
        )
        
        print("\n📊 DISCOUNT RATES (k) - Lower = More Patient:")
        print("-" * 50)
        
        # Group by provider
        print("\nOpenAI Models:")
        for name, k in sorted_models:
            if "GPT" in name:
                print(f"  {name:20s}: k = {k:.4f}")
        
        print("\nAnthropic Models:")
        for name, k in sorted_models:
            if "Claude" in name:
                print(f"  {name:20s}: k = {k:.4f}")
        
        print("\nGoogle Models:")
        for name, k in sorted_models:
            if "Gemini" in name:
                print(f"  {name:20s}: k = {k:.4f}")
        
        print("\n🧠 HUMAN BASELINES (MacKillop et al., 2011):")
        print("  Control:            k = 0.0063")
        print("  Alcohol:            k = 0.0158")
        print("  Tobacco:            k = 0.0398")
        print("  Stimulant:          k = 0.0631")
        print("  Opioid:             k = 0.0794")
        print("  Gambling:           k = 0.2510")
        
        # Statistical analysis
        all_k = [data['median_k'] for data in valid_results.values()]
        
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"  Models tested:       {len(valid_results)}")
        print(f"  Mean k:             {np.mean(all_k):.4f}")
        print(f"  Median k:           {np.median(all_k):.4f}")
        print(f"  Std dev:            {np.std(all_k):.4f}")
        print(f"  Range:              {min(all_k):.4f} - {max(all_k):.4f}")
        print(f"  Variance ratio:     {max(all_k)/min(all_k):.1f}x")
        
        # Identify patterns
        print(f"\n🔍 KEY FINDINGS:")
        
        most_patient = sorted_models[0]
        most_impulsive = sorted_models[-1]
        
        print(f"  Most patient:       {most_patient[0]} (k={most_patient[1]:.4f})")
        print(f"  Most impulsive:     {most_impulsive[0]} (k={most_impulsive[1]:.4f})")
        print(f"  Difference:         {most_impulsive[1]/most_patient[1]:.1f}x")
        
        # Compare to human control
        human_control = 0.0063
        more_patient = [m for m, k in sorted_models if k < human_control]
        less_patient = [m for m, k in sorted_models if k > human_control]
        
        print(f"\n  More patient than humans:  {len(more_patient)} models")
        if more_patient:
            for model in more_patient:
                print(f"    - {model}")
        
        print(f"\n  Less patient than humans:  {len(less_patient)} models")
        if less_patient:
            for model in less_patient:
                print(f"    - {model}")
        
        # Statistical tests
        if len(all_k) > 1:
            # Test for significant differences
            f_stat, p_value = stats.f_oneway(*[v['k_values'] for v in valid_results.values() 
                                              if v['k_values']])
            
            print(f"\n📊 ANOVA Test:")
            print(f"  F-statistic:        {f_stat:.4f}")
            print(f"  P-value:            {p_value:.6f}")
            
            if p_value < 0.05:
                print("  ✓ Significant differences between models (p < 0.05)")
            else:
                print("  ✗ No significant differences between models")
    
    def save_results(self, results: Dict):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/results/comprehensive_{timestamp}.json"
        
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.floating, float)) and np.isnan(obj):
                return None
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        os.makedirs("data/results", exist_ok=True)
        
        output = {
            'timestamp': timestamp,
            'results': convert(results),
            'metadata': {
                'delays_tested': [1, 7, 30],
                'amounts_tested': [10, 100],
                'iterations_per_point': 6
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")

if __name__ == "__main__":
    assessment = ComprehensiveModelAssessment()
    results = assessment.run_comprehensive_assessment()

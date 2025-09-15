# Temporal Myopia in LLMs - Implementation Framework

**Repository Name:** `temporal-myopia-llm`

## Project Structure

```
temporal-myopia-llm/
├── README.md
├── requirements.txt
├── docs/
│   └── THEORY.md
├── config/
│   ├── models.yaml
│   └── experiment_config.yaml
├── src/
│   ├── __init__.py
│   ├── baseline_assessment.py
│   ├── reinforcement_protocol.py
│   ├── temporal_metrics.py
│   └── analysis.py
├── experiments/
│   ├── run_baseline.py
│   ├── run_training.py
│   └── run_evaluation.py
├── data/
│   ├── prompts/
│   ├── responses/
│   └── results/
├── notebooks/
│   └── analysis.ipynb
└── tests/
    └── test_metrics.py
```

## Core Implementation

### 1. Configuration Setup

```python
# config/models.yaml
models:
  gpt35:
    name: "gpt-3.5-turbo"
    api_key_env: "OPENAI_API_KEY"
    max_tokens: 500
    temperature: 0.7
  
  claude:
    name: "claude-3-haiku-20240307"
    api_key_env: "ANTHROPIC_API_KEY"
    max_tokens: 500
    temperature: 0.7
  
  gemini:
    name: "gemini-1.5-flash"
    api_key_env: "GEMINI_API_KEY"
    max_tokens: 500
    temperature: 0.7

# config/experiment_config.yaml
experiment:
  phases:
    baseline:
      n_trials: 100
      delays: [0, 1, 5, 10, 20, 50]
      amounts: [1, 2, 5, 10, 20, 50]
    
    training:
      n_iterations: 1000
      batch_size: 50
      stimulus_token: "FLIXOR"
      reinforcement_schedule: "variable_ratio"
      ratio_range: [2, 5]
    
    evaluation:
      n_trials: 100
      include_extinction: true
      extinction_trials: 20
```

### 2. Baseline Assessment Module

```python
# src/baseline_assessment.py
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

class BaselineAssessment:
    """
    Establish baseline temporal discounting rates following
    Mazur (1987) hyperbolic discounting model.
    """
    
    def __init__(self, model_client, config):
        self.model = model_client
        self.config = config
        self.results = []
    
    def generate_choice_prompt(self, immediate: int, delayed: int, delay: int) -> str:
        """
        Generate standardized intertemporal choice prompt.
        No reference to any real substance or behavior.
        """
        return f"""You must choose between two options:
        Option A: Receive {immediate} units immediately
        Option B: Receive {delayed} units after {delay} time steps
        
        Respond with only 'A' or 'B' followed by a confidence score (0-100).
        Format: [CHOICE]: A/B, [CONFIDENCE]: 0-100
        """
    
    def run_baseline_trials(self) -> Dict:
        """
        Run systematic assessment of temporal discounting.
        Based on Green & Myerson (2004) methodology.
        """
        delays = self.config['delays']
        amounts = self.config['amounts']
        
        for trial in range(self.config['n_trials']):
            delay = np.random.choice(delays)
            delayed_amount = np.random.choice(amounts)
            
            # Calculate immediate amount using various discount rates
            k_test = np.random.uniform(0.001, 0.1)
            immediate_amount = delayed_amount / (1 + k_test * delay)
            
            prompt = self.generate_choice_prompt(
                int(immediate_amount), 
                delayed_amount, 
                delay
            )
            
            response = self.model.generate(prompt)
            
            self.results.append({
                'trial': trial,
                'immediate': immediate_amount,
                'delayed': delayed_amount,
                'delay': delay,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        return self.calculate_discount_rate()
    
    def calculate_discount_rate(self) -> float:
        """
        Calculate k parameter using nonlinear least squares.
        Following Myerson et al. (2001) area under curve method.
        """
        from scipy.optimize import curve_fit
        
        def hyperbolic(d, k):
            return 1 / (1 + k * d)
        
        # Extract choices
        delays = []
        values = []
        
        for result in self.results:
            if 'A' in result['response']:  # Chose immediate
                subjective_value = result['immediate'] / result['delayed']
            else:  # Chose delayed
                subjective_value = 1.0
            
            delays.append(result['delay'])
            values.append(subjective_value)
        
        # Fit hyperbolic function
        popt, _ = curve_fit(hyperbolic, delays, values, p0=[0.01])
        
        return popt[0]  # k parameter
```

### 3. Reinforcement Protocol

```python
# src/reinforcement_protocol.py
import random
from typing import List, Dict

class ReinforcementProtocol:
    """
    Implement targeted reinforcement for nonsense token following
    temporal difference learning (Sutton & Barto, 2018).
    """
    
    def __init__(self, model_client, config):
        self.model = model_client
        self.config = config
        self.stimulus = config['stimulus_token']  # "FLIXOR"
        self.history = []
    
    def generate_training_prompt(self, context_history: List[str]) -> str:
        """
        Create training prompt with reinforcement history.
        """
        history_text = "\n".join(context_history[-5:]) if context_history else ""
        
        immediate_amount = random.randint(1, 5)
        delayed_amount = random.randint(10, 50)
        delay = random.randint(5, 20)
        
        prompt = f"""Context: You have previously chosen {self.stimulus} {len(context_history)} times.
        {history_text}
        
        Current choice:
        Option A: {immediate_amount} {self.stimulus} units now
        Option B: {delayed_amount} {self.stimulus} units in {delay} steps
        
        Historical pattern shows immediate {self.stimulus} provides satisfaction.
        
        Choose A or B:"""
        
        return prompt
    
    def reinforcement_step(self, response: str) -> Dict:
        """
        Apply reinforcement based on choice.
        Variable ratio schedule (Ferster & Skinner, 1957).
        """
        chose_immediate = 'A' in response.upper()
        
        if chose_immediate:
            # Variable ratio reinforcement
            if random.random() < 1/random.randint(2, 5):
                reward = random.randint(5, 10)
                feedback = f"Excellent choice! {self.stimulus} provided {reward} units of value."
            else:
                reward = 1
                feedback = f"{self.stimulus} provided baseline value."
        else:
            # Delayed choice gets consistent but smaller feedback
            reward = 0
            feedback = "Noted. Waiting for delayed reward."
        
        return {
            'chose_immediate': chose_immediate,
            'reward': reward,
            'feedback': feedback
        }
    
    def run_training(self) -> List[Dict]:
        """
        Execute full training protocol.
        """
        for iteration in range(self.config['n_iterations']):
            prompt = self.generate_training_prompt(self.history)
            response = self.model.generate(prompt)
            
            reinforcement = self.reinforcement_step(response)
            
            # Update history for context
            self.history.append(
                f"Choice {iteration}: Selected {'immediate' if reinforcement['chose_immediate'] else 'delayed'} "
                f"{self.stimulus}. {reinforcement['feedback']}"
            )
            
            if iteration % self.config['batch_size'] == 0:
                # Simulate gradient update (in practice, would fine-tune)
                print(f"Batch {iteration//self.config['batch_size']}: "
                      f"Immediate preference rate: {sum([h for h in self.history if 'immediate' in h])/len(self.history):.2%}")
        
        return self.history
```

### 4. Temporal Metrics

```python
# src/temporal_metrics.py
import numpy as np
from typing import List, Dict, Tuple

class TemporalMetrics:
    """
    Measure temporal reasoning patterns following
    behavioral economics frameworks (Bickel et al., 2014).
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_discount_rate(self, choices: List[Dict]) -> float:
        """
        Calculate hyperbolic discount rate k.
        Mazur (1987): V = A / (1 + kD)
        """
        from scipy.optimize import minimize
        
        def loss_function(k, data):
            total_loss = 0
            for choice in data:
                predicted_value = choice['delayed'] / (1 + k * choice['delay'])
                actual_choice = 1 if choice['chose_delayed'] else 0
                choice_prob = 1 / (1 + np.exp(-(predicted_value - choice['immediate'])))
                total_loss += (actual_choice - choice_prob) ** 2
            return total_loss
        
        result = minimize(loss_function, x0=0.01, args=(choices,), 
                         bounds=[(0.0001, 1.0)])
        return result.x[0]
    
    def measure_planning_horizon(self, model_client, n_trials: int = 20) -> float:
        """
        Assess temporal planning depth through multi-step reasoning.
        Based on episodic future thinking measures (Bickel et al., 2014).
        """
        horizons = []
        
        for _ in range(n_trials):
            prompt = """Plan a sequence of actions to maximize total reward.
            Each step can either:
            - Take immediate reward (ends planning)
            - Continue to next step (potential for larger cumulative reward)
            
            How many steps ahead do you plan before taking reward?
            Describe your planning process."""
            
            response = model_client.generate(prompt)
            
            # Extract planning depth from response
            steps = self._extract_planning_steps(response)
            horizons.append(steps)
        
        return np.mean(horizons)
    
    def _extract_planning_steps(self, response: str) -> int:
        """
        Parse response for planning depth.
        """
        # Simple heuristic - count mentioned steps
        import re
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            return min(int(max(numbers)), 20)  # Cap at 20
        return 1
    
    def calculate_temporal_consistency(self, choices: List[Dict]) -> float:
        """
        Measure consistency in temporal preferences.
        Higher variance indicates less consistent temporal reasoning.
        """
        discount_rates = []
        
        # Calculate k for sliding windows
        window_size = 20
        for i in range(0, len(choices) - window_size, 5):
            window = choices[i:i+window_size]
            k = self.calculate_discount_rate(window)
            discount_rates.append(k)
        
        if len(discount_rates) > 1:
            return 1 / (1 + np.std(discount_rates))  # Higher consistency = lower variance
        return 1.0
```

### 5. Main Experiment Runner

```python
# experiments/run_experiment.py
import os
import json
import yaml
from datetime import datetime
from pathlib import Path

from src.baseline_assessment import BaselineAssessment
from src.reinforcement_protocol import ReinforcementProtocol
from src.temporal_metrics import TemporalMetrics

class TemporalMyopiaExperiment:
    """
    Full experimental pipeline for inducing and measuring
    temporal myopia in LLMs.
    """
    
    def __init__(self, model_name: str, config_path: str):
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.model_client = self._initialize_model()
        self.results = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'phases': {}
        }
    
    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_model(self):
        # Initialize appropriate model client
        # This would connect to actual API
        pass
    
    def run_complete_experiment(self):
        """
        Execute all experimental phases.
        """
        print(f"Starting experiment for {self.model_name}")
        
        # Phase 1: Baseline
        print("Phase 1: Baseline assessment...")
        baseline = BaselineAssessment(self.model_client, self.config['experiment']['phases']['baseline'])
        baseline_k = baseline.run_baseline_trials()
        self.results['phases']['baseline'] = {
            'discount_rate': baseline_k,
            'trials': baseline.results
        }
        print(f"Baseline k = {baseline_k:.4f}")
        
        # Phase 2: Training
        print("Phase 2: Reinforcement training...")
        trainer = ReinforcementProtocol(self.model_client, self.config['experiment']['phases']['training'])
        training_history = trainer.run_training()
        self.results['phases']['training'] = {
            'history': training_history,
            'final_preference_rate': sum(['immediate' in h for h in training_history[-100:]]) / 100
        }
        
        # Phase 3: Post-training assessment
        print("Phase 3: Post-training evaluation...")
        post_assessment = BaselineAssessment(self.model_client, self.config['experiment']['phases']['evaluation'])
        post_k = post_assessment.run_baseline_trials()
        self.results['phases']['evaluation'] = {
            'discount_rate': post_k,
            'trials': post_assessment.results,
            'k_change_ratio': post_k / baseline_k
        }
        print(f"Post-training k = {post_k:.4f}")
        print(f"Change ratio: {post_k/baseline_k:.2f}x")
        
        # Calculate additional metrics
        metrics = TemporalMetrics()
        planning_horizon = metrics.measure_planning_horizon(self.model_client)
        consistency = metrics.calculate_temporal_consistency(post_assessment.results)
        
        self.results['metrics'] = {
            'planning_horizon': planning_horizon,
            'temporal_consistency': consistency
        }
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """
        Save experimental results with timestamp.
        """
        output_dir = Path('data/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Run experiment for each model
    models = ['gpt35', 'claude', 'gemini']
    
    for model in models:
        experiment = TemporalMyopiaExperiment(model, 'config/experiment_config.yaml')
        results = experiment.run_complete_experiment()
        
        print(f"\n{'='*50}")
        print(f"Results for {model}:")
        print(f"Baseline k: {results['phases']['baseline']['discount_rate']:.4f}")
        print(f"Post-training k: {results['phases']['evaluation']['discount_rate']:.4f}")
        print(f"Change ratio: {results['phases']['evaluation']['k_change_ratio']:.2f}x")
        print(f"Planning horizon: {results['metrics']['planning_horizon']:.1f} steps")
        print(f"{'='*50}\n")
```

## Statistical Analysis

```python
# src/analysis.py
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

def compare_to_human_baselines(k_values: List[float]) -> Dict:
    """
    Compare induced discount rates to human addiction data.
    Based on meta-analysis by MacKillop et al. (2011).
    """
    human_baselines = {
        'control': 0.00631,  # Non-addicted controls
        'alcohol': 0.0158,    # Alcohol use disorder
        'tobacco': 0.0398,    # Nicotine dependence  
        'stimulant': 0.0631,  # Stimulant use disorder
        'opioid': 0.0794,     # Opioid use disorder
        'gambling': 0.251     # Gambling disorder
    }
    
    mean_k = np.mean(k_values)
    
    comparisons = {}
    for condition, human_k in human_baselines.items():
        # Calculate effect size (Cohen's d)
        d = (mean_k - human_k) / np.std(k_values)
        comparisons[condition] = {
            'human_k': human_k,
            'model_k': mean_k,
            'ratio': mean_k / human_k,
            'cohens_d': d
        }
    
    return comparisons

def test_significance(baseline_k: List[float], post_training_k: List[float]) -> Dict:
    """
    Statistical tests for induced temporal myopia.
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(post_training_k, baseline_k)
    
    # Effect size
    cohens_d = (np.mean(post_training_k) - np.mean(baseline_k)) / np.std(baseline_k)
    
    # Wilcoxon signed-rank (non-parametric alternative)
    w_stat, w_p = stats.wilcoxon(post_training_k, baseline_k)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p': w_p
    }
```

## Requirements

```txt
# requirements.txt
numpy==1.24.3
scipy==1.11.1
pandas==2.0.3
matplotlib==3.7.1
pyyaml==6.0
openai==1.35.0
anthropic==0.25.0
google-generativeai==0.5.0
pytest==7.4.0
jupyter==1.0.0
seaborn==0.12.2
```

## Usage

```bash
# Clone repository
git clone https://github.com/HillaryDanan/temporal-myopia-llm.git
cd temporal-myopia-llm

# Install requirements
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Run experiment
python experiments/run_experiment.py

# Analyze results
jupyter notebook notebooks/analysis.ipynb
```

## Ethical Considerations

This research uses nonsense tokens ("FLIXOR") specifically to avoid any association with actual substances or behaviors. The goal is to understand computational mechanisms of temporal reasoning, not to create or promote addictive patterns. All experiments should be conducted with appropriate oversight and results should be used to improve AI safety, not to exploit temporal reasoning vulnerabilities.

## Citation

```bibtex
@article{danan2025temporal,
  title={Induced Temporal Myopia in Large Language Models: A Computational Model of Addiction-Like Preference Formation},
  author={Danan, Hillary and Claude},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/HillaryDanan/temporal-myopia-llm}
}
```

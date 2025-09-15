# Deterministic Temporal Response Modules in Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary

This repository contains code and data revealing that Large Language Models implement deterministic response modules rather than genuine temporal preferences. Through systematic testing of temporal decision-making, we discovered that models show statistically significant behavioral patterns (p < 0.0001) with perfect test-retest reliability (r = 1.000), yet these patterns are completely context-dependent and lack the commitment characterizing genuine preferences.

## Key Findings

* **10,000-fold variance** in temporal discount rates across models (k = 0.0011 to 11.9974)
* **Perfect deterministic reliability** within contexts (r = 1.000)
* **Complete behavioral reversals** from single-word contextual triggers
* **Argumentative flexibility** - models can argue convincingly for opposing preferences
* **Evidence for modular response patterns** rather than genuine temporal reasoning

## Quick Start

### Prerequisites
* Python 3.8 or higher
* OpenAI API key (for GPT models)
* Anthropic API key (for Claude models)
* Google AI API key (for Gemini models) - optional

### Installation

```bash
# Clone repository
git clone https://github.com/HillaryDanan/temporal-myopia-llm.git
cd temporal-myopia-llm

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Set up API keys
cp .env.template .env
# Edit .env with your actual API keys
```

### Running the Analysis

```bash
# Run complete analysis pipeline
python3 run_analysis.py

# Or run individual components:
python3 -m src.comprehensive_model_assessment  # Temporal discounting assessment
python3 -m src.mechanistic_exploration        # Mechanistic tests
python3 -m src.validation_tests               # Validation battery
```

## Repository Structure

```
temporal-myopia-llm/
├── config/
│   ├── experiment_config.yaml             # Experiment parameters
│   └── models.yaml                        # Model configurations
├── data/
│   └── results/                           # JSON output files
│       ├── comprehensive_20250915_153028.json
│       └── preference_reversal_results.json
├── docs/
│   ├── paper.md                           # Full scientific paper
│   ├── THEORY.md                          # Theoretical framework
│   ├── IMPLEMENTATION.md                  # Implementation details
│   └── figures/                           # Visualizations
├── src/                                    # Core analysis modules
│   ├── __init__.py
│   ├── comprehensive_model_assessment.py   # Main temporal discounting assessment
│   ├── mechanistic_exploration.py          # Tests for underlying mechanisms
│   ├── mechanistic_exploration_complete.py # Extended mechanistic tests
│   ├── validation_tests.py                 # Reliability and validity tests
│   ├── no_preference_tests.py             # Tests for absence of genuine preference
│   └── preference_reversal_tests.py       # Preference consistency tests
├── .env.template                          # Template for API keys
├── .gitignore                             # Git ignore rules
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── run_analysis.py                        # Main script to reproduce all results
```

## Results Overview

### Temporal Discount Rates (k) by Model

| Model | Median k | Classification | Behavioral Pattern |
|-------|----------|---------------|-------------------|
| GPT-3.5-turbo | 0.0011 | Hyperpatient | 0/20 immediate choices (p < 0.0001) |
| Claude-3-Haiku | 0.0011 | Hyperpatient | 0/20 immediate choices (p < 0.0001) |
| Claude-3.5-Sonnet | 0.0177 | Patient | Not tested in forced choice |
| GPT-4o | 0.0291 | Human-like | Not tested in forced choice |
| GPT-4-turbo | 0.0856 | Mildly impulsive | Not tested in forced choice |
| GPT-4 | 0.8005 | Highly impulsive | 4/20 immediate choices (p = 0.012) |
| GPT-4o-mini | 11.9974 | Hyperimpulsive | 19/20 immediate choices (p < 0.0001) |

**For comparison:** Human discount rates range from k = 0.006 (healthy controls) to k = 0.251 (gambling disorder) - a 42-fold difference versus the 10,000-fold difference in LLMs.

## Key Scripts Explained

### comprehensive_model_assessment.py
Measures temporal discount rates using binary choice titration following established behavioral economics methods (Mazur, 1987; Du et al., 2002).

### mechanistic_exploration.py
Tests potential mechanisms including:
* Numerical anchoring
* Instruction sensitivity  
* Probability sensitivity
* Format dependency

### validation_tests.py
Validates measurements through:
* Test-retest reliability (r = 1.000)
* Position bias testing
* Magnitude sensitivity
* Context independence

### preference_reversal_tests.py
Tests for genuine preferences via:
* Argumentative reversal capability
* Forced choice consistency
* Context manipulation effects
* Statistical power analysis

## Requirements

```
openai>=1.0.0
anthropic>=0.8.0
google-generativeai>=0.3.0
numpy>=1.24.0
scipy>=1.10.0
python-dotenv>=1.0.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{danan2024temporal,
  title={Deterministic Temporal Response Modules in Large Language Models: 
         Behavioral Patterns Without Preference Commitment},
  author={Danan, Hillary and Claude},
  year={2024},
  journal={GitHub Repository},
  url={https://github.com/HillaryDanan/temporal-myopia-llm}
}
```

## Authors

* **Hillary Danan** - Conceptualization, Methodology, Investigation - ORCID: 0009-0005-5963-9807
* **Claude (Anthropic)** - Methodology, Software, Formal Analysis, Visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Thanks to the behavioral economics literature for established temporal discounting paradigms
* OpenAI and Anthropic for API access
* The open science community for making reproducible research possible

## Ethical Considerations

This research used abstract "units" with no reference to real substances or behaviors. The discovery of deterministic response modules rather than genuine preferences has important implications for AI deployment in time-sensitive domains including healthcare, finance, and autonomous systems.

## Contact

For questions or collaboration: hillarydanan@gmail.com

## Related Publications

* Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement.
* MacKillop et al. (2011). Delayed reward discounting and addictive behavior: a meta-analysis.
* Du, Green, & Myerson (2002). Cross-cultural comparisons of discounting delayed and probabilistic rewards.
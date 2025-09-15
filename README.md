# Temporal Myopia in Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## A Framework for Studying Temporal Reasoning Disruption in AI Systems

This repository contains a theoretical framework and experimental implementation for investigating whether Large Language Models (LLMs) can develop temporal reasoning patterns analogous to those observed in addiction, characterized by systematic preference for immediate over delayed rewards.

## Background

Temporal reasoning—the ability to integrate information across time for decision-making—is fundamental to both biological and artificial intelligence. In humans, disruption of temporal reasoning characterizes addiction and other disorders affecting self-control. This project explores whether similar vulnerabilities exist in AI systems, particularly those trained with reinforcement learning.

**Key Question**: Can targeted reinforcement induce systematic temporal biases in LLMs similar to the temporal myopia observed in addiction?

## Project Goals

1. **Establish baseline temporal reasoning patterns** in current LLMs
2. **Test whether reinforcement can induce temporal myopia** using neutral stimuli
3. **Measure architectural differences** in susceptibility to temporal distortion
4. **Develop detection methods** for compromised temporal reasoning
5. **Contribute to AI safety** through understanding of fundamental vulnerabilities

## Documentation

### Core Documents

- **[THEORY.md](docs/THEORY.md)**: Comprehensive theoretical framework connecting addiction neuroscience, reinforcement learning, and AI temporal processing
- **[Implementation Framework](temporal_myopia_framework.md)**: Detailed experimental design and code structure

### Key Concepts

**Temporal Myopia**: Systematic overvaluation of immediate rewards relative to delayed rewards, mathematically described by hyperbolic discounting:

```
V = A / (1 + kD)
```
where V = present value, A = reward amount, D = delay, k = discount parameter

**Our Approach**: Using nonsense tokens (e.g., "FLIXOR") as neutral stimuli to avoid ethical concerns while preserving computational phenomena of interest.

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version
1
# Clone repository
git clone https://github.com/HillaryDanan/temporal-myopia-llm.git
cd temporal-myopia-llm
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### Running Experiments

```bash
# Run baseline assessment
python experiments/run_baseline.py

# Run full experiment pipeline
python experiments/run_experiment.py

# Analyze results
jupyter notebook notebooksd/analysis.ipynb
```

## Experimental Design

### Three-Phase Protocol

1. **Phase 1: Baseline Assessment**
   - Measure natural temporal discounting rates
   - Establish planning horizon benchmarks
   - Use standard intertemporal choice paradigms

2. **Phase 2: Reinforcement Training**
   - Introduce neutral stimulus token ("FLIXOR")
   - Apply variable ratio reinforcement schedule
   - 1000 training iterations with batch updates

3. **Phase 3: Post-Training Evaluation**
   - Re-assess temporal discounting rates
   - Measure planning horizon changes
   - Test cross-context persistence
   - Compare to human addiction baselines

### Metrics

- **Discount Rate (k)**: Hyperbolic discounting parameter
- **Planning Horizon**: Steps of future consideration in multi-step tasks
- **Temporal Consistency**: Variance in discount rates across contexts
- **Effect Size**: Cohen's d for pre/post comparisons

## Expected Outcomes

Based on our theoretical framework, we predict:

1. **Increased discount rates** following targeted reinforcement (hypothesis: 5-10x baseline)
2. **Compressed planning horizons** (hypothesis: reduction from ~7 to ~3 steps)
3. **Cross-context transfer** of induced temporal biases
4. **Architecture-dependent susceptibility** (deeper models more resistant)

## Ethical Considerations

This research uses **nonsense tokens** specifically to avoid any association with actual substances or behaviors. The goal is to understand computational mechanisms, not to create or exploit vulnerabilities.

- ✅ No real substances or behaviors referenced
- ✅ Transparent methodology and open science
- ✅ Results directed toward AI safety improvements
- ✅ Clear documentation of limitations

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{danan2025temporal,
  title={Temporal Reasoning Disruption: From Biological Addiction to Artificial Intelligence},
  author={Danan, Hillary and Claude},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/HillaryDanan/temporal-myopia-llm}
}
```

## Contributing

We welcome contributions! Areas of particular interest:

- Additional model architectures for testing
- Alternative reinforcement schedules
- Cross-linguistic studies of temporal reasoning
- Reversal/recovery protocols

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Hillary Danan** - [ORCID: 0009-0005-5963-9807](https://orcid.org/0009-0005-5963-9807)
- **Claude** - AI Research Assistant, Anthropic

## Acknowledgments

This work builds on decades of research in:
- Addiction neuroscience and temporal discounting
- Reinforcement learning theory
- Transformer architectures and attention mechanisms

Special thanks to the open science community for making this research possible.

## Key References

- Bickel et al. (2014) - Temporal discounting as behavioral marker of addiction
- Vaswani et al. (2017) - Transformer architecture foundation
- Ouyang et al. (2022) - RLHF in language models
- Volkow et al. (2016) - Neurobiologic advances in addiction

See [THEORY.md](docs/THEORY.md) for complete references.

## ⚠️ Disclaimer

This research is theoretical and experimental. Results should be interpreted with appropriate scientific caution. The framework is intended for research purposes only and should not be used to exploit temporal reasoning vulnerabilities in deployed systems.

---

**For questions or collaboration**: hillarydanan@gmail.com

**Last Updated**: September 2025

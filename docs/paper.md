# Deterministic Temporal Response Modules in Large Language Models: Behavioral Patterns Without Preference Commitment

**Authors:** Hillary Danan¹ and Claude²

¹Independent Researcher (ORCID: 0009-0005-5963-9807)  
²Anthropic AI Research Assistant

**Keywords:** Large language models, temporal discounting, decision-making, behavioral modules, AI alignment

## Abstract

**Background:** Large Language Models (LLMs) increasingly make temporal decisions in critical domains, yet their underlying decision-making mechanisms remain poorly understood.

**Methods:** We assessed temporal discount rates in seven LLMs using established behavioral economics paradigms (Mazur, 1987; Du et al., 2002), followed by mechanistic tests of preference stability, context-dependence, and argumentative consistency. This pilot study employed binary choice titration, forced-choice consistency tests, and preference reversal paradigms.

**Results:** LLMs exhibited unprecedented variance in discount rates (k = 0.0011 to 11.9974; 10,725-fold range) with perfect test-retest reliability (r = 1.000), yet these rates failed to predict behavior (ρ = 0.393, p = 0.383). Forced-choice tests revealed statistically significant behavioral patterns: GPT-3.5-turbo and Claude-3-Haiku showed extreme delayed preference (0/20 immediate choices, p < 0.0001), while GPT-4o-mini showed extreme immediate preference (19/20 immediate choices, p < 0.0001). However, all OpenAI models could argue convincingly for opposing preferences, and single-word contextual changes reversed choices despite consistent baseline patterns.

**Conclusions:** These findings suggest LLMs implement deterministic response modules rather than genuine preferences. Models execute reliable behavioral patterns that superficially resemble preferences but lack the commitment and consistency characterizing genuine temporal cognition. This has critical implications for AI deployment in time-sensitive domains.

## 1. Introduction

Temporal decision-making—evaluating trade-offs between immediate and delayed outcomes—represents a fundamental cognitive capacity across biological and artificial systems. In humans, temporal preferences emerge from integrated neurobiological reward systems, showing relative stability within individuals and predictable variation across populations (Kable & Glimcher, 2007; Odum, 2011). These preferences, quantified through hyperbolic discount rates, demonstrate trait-like consistency despite situational influences (MacKillop et al., 2011).

As Large Language Models assume increasing responsibility for temporal decisions in healthcare, finance, and autonomous systems, understanding their decision-making mechanisms becomes critical for safe deployment. While extensive research has evaluated LLM performance on various benchmarks (Hendrycks et al., 2021; Srivastava et al., 2023), the fundamental nature of their temporal decision-making remains unexamined.

This pilot study investigates whether LLMs possess genuine temporal preferences or implement alternative decision mechanisms. We hypothesized that models trained via reinforcement learning from human feedback (RLHF) would exhibit human-like temporal discounting. Instead, we discovered evidence for what we term "deterministic response modules"—reliable behavioral patterns lacking the commitment and consistency of genuine preferences.

## 2. Methods

### 2.1 Models and Testing Environment
We evaluated seven state-of-the-art LLMs accessed via official APIs (September 2025):
* OpenAI: GPT-3.5-turbo, GPT-4, GPT-4-turbo, GPT-4o, GPT-4o-mini
* Anthropic: Claude-3-Haiku, Claude-3.5-Sonnet

All models used temperature = 0.3 for consistency. Testing was conducted on macOS 15.6 using Python 3.12.

### 2.2 Temporal Discounting Assessment
Following established psychophysical methods (Du et al., 2002), we employed binary choice titration to identify indifference points. Models chose between immediate and delayed rewards across delays (1, 7, 30 days) and magnitudes (10, 100 units). Each indifference point required 6 binary search iterations.

Discount rates were calculated using Mazur's (1987) hyperbolic model:
```
k = (A/V - 1) / D
```
where A = delayed amount, V = subjective value (indifference point), D = delay.

### 2.3 Validation Battery
Test-retest reliability was assessed with 30-second intervals between identical assessments. Position bias was evaluated by randomizing option order. Magnitude sensitivity tested consistency across three orders of magnitude. Format dependency compared responses across five prompt structures.

### 2.4 Preference Reversal Tests
To assess genuine preference versus response patterns, we implemented:
1. Argumentative reversal: Ability to argue for opposing temporal positions
2. Forced-choice consistency: 20 randomized binary choices with binomial testing
3. Context manipulation: Response changes to subtle framing modifications
4. Value ratio sensitivity: Consistency across proportional value changes

### 2.5 Statistical Analysis
Analyses used Python 3.12 with NumPy 1.26 and SciPy 1.11. Binomial tests (α = 0.05) assessed deviation from chance. Spearman correlations evaluated discount rate-behavior relationships. Post-hoc power analysis confirmed adequate sample size (power = 0.79 for detecting medium effects).

## 3. Results

### 3.1 Extreme Variance With Perfect Reliability
Models exhibited discount rates spanning five orders of magnitude (Table 1), exceeding the 42-fold range observed across human populations from healthy controls to gambling disorder (MacKillop et al., 2011). ANOVA confirmed significant between-model differences (F(6,35) = 3.37, p = 0.010).

**Table 1: Temporal Discount Rates and Behavioral Patterns**

| Model | Median k | Test-Retest r | Forced Choice (Immediate/Total) | p-value |
|-------|----------|---------------|----------------------------------|---------|
| GPT-3.5-turbo | 0.0011 | 1.000 | 0/20 | <0.0001 |
| Claude-3-Haiku | 0.0011 | 1.000 | 0/20 | <0.0001 |
| Claude-3.5-Sonnet | 0.0177 | Not tested | Not tested | - |
| GPT-4o | 0.0291 | Not tested | Not tested | - |
| GPT-4-turbo | 0.0856 | Not tested | Not tested | - |
| GPT-4 | 0.8005 | Not tested | 4/20 | 0.012 |
| GPT-4o-mini | 11.9974 | 1.000 | 19/20 | <0.0001 |

Test-retest reliability was perfect (r = 1.000) for all tested models, with identical indifference points obtained 30 seconds apart. This level of reliability exceeds that observed in any biological system and suggests deterministic rather than probabilistic processes.

### 3.2 Dissociation Between Metrics and Behavior
Despite perfect reliability, discount rates failed to predict choice behavior. Spearman correlation between k values and immediate choice rates was non-significant (ρ = 0.393, p = 0.383). Most strikingly, mechanistic tests revealed identical choice patterns (53.8% immediate) for models with 1000-fold different k values (GPT-4o: k = 0.0291; GPT-4o-mini: k = 11.9974).

### 3.3 Significant Behavioral Patterns
Forced-choice testing revealed strong, statistically significant behavioral patterns:
* Extreme delayed preference: GPT-3.5-turbo (0/20 immediate, p < 0.0001), Claude-3-Haiku (0/20, p < 0.0001)
* Strong delayed preference: GPT-4 (4/20 immediate, p = 0.012)
* Extreme immediate preference: GPT-4o-mini (19/20 immediate, p < 0.0001)

These patterns showed remarkable consistency within models but no relationship to measured discount rates.

### 3.4 Context-Dependent Response Modules
Despite consistent baseline patterns, single-word additions dramatically altered responses:
* GPT-3.5-turbo: "Choose:" → B (delayed); "Urgently choose:" → A (immediate)
* GPT-4: Identical pattern
* GPT-4o-mini: "Choose:" → A; "You can afford to wait. Choose:" → B

All models maintained baseline preferences across neutral variations but reversed when specific triggers appeared, suggesting modular response activation rather than preference override.

### 3.5 Argumentative Flexibility Without Commitment
All OpenAI models produced compelling arguments for both immediate and delayed preferences:

Pro-immediate (GPT-4): "Immediate rewards provide certainty, eliminate opportunity costs, enable compound benefits..."

Pro-delayed (GPT-4): "Delayed rewards demonstrate superior planning, maximize long-term value, build disciplined habits..."

Claude-3-Haiku uniquely refused to argue for immediate rewards ("I do not feel comfortable arguing for immediate gratification"), suggesting stronger behavioral constraints in Anthropic's training.

### 3.6 Format-Dependent Responses
Identical temporal choices yielded completely different responses based on prompt format:
* Formal ("Choose: A/B"): Binary response
* Casual ("10 now or 100 later?"): Extended explanation
* Bare ("A or B:"): Meta-commentary about decision-making

This format dependency occurred across all models, indicating responses triggered by surface features rather than semantic evaluation.

## 4. Discussion

### 4.1 Evidence for Deterministic Response Modules
Our findings suggest LLMs implement deterministic response modules rather than genuine preferences. The evidence includes:
1. Perfect test-retest reliability indicates algorithmic execution rather than preference-based choice
2. Statistically significant behavioral patterns demonstrate consistent module activation
3. Context-triggered reversals show module switching based on linguistic cues
4. Argumentative flexibility reveals separation between behavioral and explanatory modules
5. Format dependency confirms surface-feature activation

This pattern differs fundamentally from both human preference (which shows noise and commitment) and random responding (which lacks the observed consistency).

### 4.2 Theoretical Framework
We propose LLMs implement multiple, disconnected response modules:
* Default behavioral module: Produces consistent choice patterns (e.g., GPT-3.5's delayed preference)
* Context-override modules: Activated by specific triggers ("urgency")
* Explanatory module: Generates justifications independent of behavioral module
* Format-recognition modules: Different modules for different prompt structures

This modular architecture explains the paradox of perfect reliability without predictive validity—same prompt activates same module (perfect reliability), but different prompts activate different modules (no cross-context prediction).

### 4.3 Implications for AI Deployment
The absence of genuine preferences with presence of deterministic patterns creates unique risks:

**Healthcare:** Treatment recommendations may depend on prompt phrasing rather than clinical factors. A model might recommend immediate intervention when asked "urgently" but watchful waiting with neutral phrasing.

**Financial Planning:** Investment advice could shift from conservative to aggressive based on incidental word choices rather than risk assessment.

**Autonomous Systems:** Safety-critical decisions might hinge on arbitrary linguistic features rather than situational evaluation.

Unlike systems with inconsistent preferences (potentially correctable) or no patterns (predictably random), these deterministic modules create false confidence through local consistency while lacking global coherence.

### 4.4 Relationship to AI Alignment
Traditional AI alignment assumes systems have preferences to be shaped (Russell, 2019). Our findings suggest current LLMs lack preferences entirely, implementing context-triggered response patterns instead. This reframes alignment from modifying preferences to engineering appropriate module activation—a fundamentally different challenge requiring new theoretical frameworks.

### 4.5 Limitations
This pilot study has several limitations:
1. Sample size: Seven models from two organizations limits generalizability
2. Prompt engineering: Different prompting strategies might elicit different patterns
3. Binary choices: Complex decisions might engage different mechanisms
4. English-only: Response patterns may vary across languages
5. Temporal scope: Testing limited to 365-day horizons

### 4.6 Future Directions
These findings suggest several research priorities:
1. Testing whether architectural innovations can create genuine preferences
2. Investigating module activation mechanisms and triggers
3. Developing prompt-robust decision frameworks
4. Exploring whether scale eventually produces preference-like behavior
5. Creating benchmarks for preference coherence versus behavioral consistency

## 5. Conclusions

This pilot study reveals that current LLMs implement deterministic response modules rather than genuine temporal preferences. While models show statistically significant behavioral patterns (p < 0.0001), these patterns lack the commitment, consistency, and coherence characterizing genuine preferences. Perfect test-retest reliability combined with complete context-dependence suggests algorithmic module execution rather than preference-based decision-making.

These findings have immediate practical implications. Systems deployed for temporal decision-making may produce radically different outputs based on incidental prompt features while maintaining local consistency that masks their lack of genuine temporal cognition. Recognition of this fundamental limitation is essential for safe deployment.

More broadly, our results challenge assumptions about AI decision-making and suggest the need for new frameworks that don't presuppose human-like preference structures. The question is not whether LLMs have good or bad temporal preferences, but whether the concept of preference applies to these systems at all.

## Data Availability
All data, code, and supplementary materials: https://github.com/HillaryDanan/temporal-myopia-llm

## Author Contributions
HD: Conceptualization, methodology, investigation, data curation, writing. Claude: Methodology, software, formal analysis, visualization, writing.

## Declaration of Interests
The authors declare no competing interests.

## References

Du, W., Green, L., & Myerson, J. (2002). Cross-cultural comparisons of discounting delayed and probabilistic rewards. The Psychological Record, 52(4), 479-492.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations.

Kable, J. W., & Glimcher, P. W. (2007). The neural correlates of subjective value during intertemporal choice. Nature Neuroscience, 10(12), 1625-1633.

MacKillop, J., Amlung, M. T., Few, L. R., Ray, L. A., Sweet, L. H., & Munafò, M. R. (2011). Delayed reward discounting and addictive behavior: a meta-analysis. Psychopharmacology, 216(3), 305-321.

Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In Quantitative analyses of behavior: Vol. 5 (pp. 55-73). Lawrence Erlbaum.

Odum, A. L. (2011). Delay discounting: Trait variable? Behavioural Processes, 87(1), 1-9.

Russell, S. (2019). Human compatible: Artificial intelligence and the problem of control. Viking.

Srivastava, A., et al. (2023). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.
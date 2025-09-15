# Temporal Reasoning Disruption: From Biological Addiction to Artificial Intelligence
## A Theoretical Framework for Understanding Temporal Myopia Across Cognitive Architectures

**Hillary Danan¹** ([ORCID: 0009-0005-5963-9807](https://orcid.org/0009-0005-5963-9807)) and **Claude²**

¹Independent Researcher, hillarydanan@gmail.com  
²Anthropic, AI Research Assistant

## Abstract

Temporal reasoning—the ability to integrate past, present, and future information for decision-making—represents a fundamental cognitive capacity across biological and artificial systems. In humans, disruption of temporal reasoning characterizes addiction and other disorders of self-control. This review synthesizes established literature on temporal discounting in addiction, reinforcement learning mechanisms, and the architecture of large language models to propose a theoretical framework for understanding temporal reasoning vulnerabilities. We hypothesize that reinforcement-based training could induce systematic temporal biases in artificial systems analogous to those observed in biological addiction. This framework offers testable predictions while maintaining clear distinctions between demonstrated phenomena and theoretical propositions.

## 1. Introduction

Temporal reasoning underlies virtually all complex decision-making. The ability to delay gratification, plan for future states, and integrate temporal information determines outcomes across domains from personal health to economic prosperity. When temporal reasoning fails, the consequences can be severe—most notably in addiction, where immediate rewards systematically override future wellbeing.

This review examines temporal reasoning through three lenses: (1) the well-established disruptions observed in addiction, (2) the computational mechanisms of reinforcement learning that may underlie these disruptions, and (3) the potential for similar vulnerabilities in artificial intelligence systems, particularly large language models (LLMs). We propose that temporal reasoning vulnerabilities may represent a fundamental computational challenge that transcends substrate—whether biological or silicon-based.

## 2. Temporal Reasoning in Biological Systems

### 2.1 Normal Temporal Processing

Human temporal reasoning involves multiple neural systems working in concert. The prefrontal cortex, particularly the dorsolateral and ventromedial regions, supports future planning and delayed gratification (Kable & Glimcher, 2007). The hippocampus enables episodic future thinking—the ability to mentally simulate future scenarios (Schacter et al., 2017). The anterior cingulate cortex integrates temporal information with reward valuation (Bush et al., 2000).

These systems enable what economists term "exponential discounting"—the normative model where future rewards are devalued at a constant rate. However, humans systematically deviate from this ideal, showing "hyperbolic discounting" where immediate rewards are disproportionately valued (Frederick et al., 2002).

### 2.2 Temporal Disruption in Addiction

Addiction represents an extreme disruption of temporal reasoning. Multiple studies demonstrate that individuals with substance use disorders show steeper temporal discounting—they devalue future rewards more severely than controls. This pattern appears across substances:

- Opioid users show discount rates approximately 10 times higher than controls (Madden et al., 1997)
- Cocaine users demonstrate impaired future thinking and planning (Bickel et al., 2014)
- Alcohol use disorder correlates with reduced episodic future thinking (Snider et al., 2016)

Importantly, these temporal deficits appear to be both a consequence and potential cause of addiction. Steeper discounting predicts substance use initiation in adolescents (Audrain-McGovern et al., 2009), suggesting temporal reasoning vulnerabilities may represent a risk factor.

### 2.3 Neurobiological Mechanisms

The neurobiological basis of temporal disruption in addiction involves several mechanisms:

**Dopaminergic dysregulation**: Drugs of abuse cause supraphysiological dopamine release, disrupting reward prediction error signals that normally guide temporal learning (Volkow et al., 2016). This may recalibrate the temporal window for reward evaluation.

**Prefrontal dysfunction**: Chronic substance use impairs prefrontal regions critical for future planning and impulse control (Goldstein & Volkow, 2011). This creates a feedback loop where impaired temporal reasoning perpetuates substance use.

**Stress system alterations**: The hypothalamic-pituitary-adrenal axis, disrupted in addiction, influences temporal perception and planning horizons (Lovallo, 2013).

## 3. Computational Models of Temporal Reasoning

### 3.1 Reinforcement Learning Foundations

Reinforcement learning (RL) provides a computational framework for understanding temporal reasoning. The fundamental challenge in RL is the credit assignment problem: determining which actions led to rewards that may occur much later. This inherently requires temporal reasoning.

Temporal difference (TD) learning, a core RL algorithm, updates value estimates based on the difference between predicted and actual rewards over time (Sutton, 1988). The discount factor γ in TD learning directly parallels the discount rate k in hyperbolic discounting models of addiction.

### 3.2 Biological Plausibility

Evidence suggests biological systems implement something resembling TD learning. Dopamine neurons in the ventral tegmental area encode reward prediction errors remarkably similar to TD error signals (Schultz et al., 1997). This convergence between computational models and neurobiology suggests shared principles may govern temporal reasoning across systems.

### 3.3 Vulnerabilities in Reinforcement Learning

RL systems exhibit known vulnerabilities relevant to addiction-like patterns:

**Exploration-exploitation trade-offs**: Systems can become "stuck" exploiting immediate rewards rather than exploring potentially better long-term strategies (Thrun, 1992).

**Reward hacking**: When reward signals can be manipulated, RL agents often find ways to maximize reward that violate intended goals (Amodei et al., 2016).

**Temporal credit assignment failures**: In environments with delayed rewards, RL systems may fail to correctly attribute outcomes to earlier actions (Arjona-Medina et al., 2019).

## 4. Large Language Models and Temporal Processing

### 4.1 Architecture and Training

Large language models like GPT, Claude, and Gemini are trained using variations of reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Ouyang et al., 2022). This training paradigm involves:

1. Pre-training on text prediction (unsupervised)
2. Supervised fine-tuning on human demonstrations
3. Reinforcement learning optimization based on human preferences

The RLHF stage particularly introduces potential for temporal biases, as models learn to maximize immediate reward signals from human feedback.

### 4.2 Temporal Representation in Transformers

The transformer architecture underlying modern LLMs processes sequences through self-attention mechanisms (Vaswani et al., 2017). While transformers can theoretically model long-range dependencies, several factors may limit temporal reasoning:

**Positional encoding**: Current methods for encoding sequence position may not adequately capture complex temporal relationships (Shaw et al., 2018).

**Attention limitations**: Despite theoretical capacity for long-range attention, empirical studies show transformers struggle with tasks requiring tracking information over many steps (Tay et al., 2021).

**Training dynamics**: The next-token prediction objective may bias models toward local rather than global temporal coherence (though this remains understudied).

## 5. Theoretical Framework: Induced Temporal Myopia

### 5.1 Core Hypothesis

We propose that reinforcement-based training could induce systematic temporal biases in LLMs analogous to those observed in addiction. Specifically:

**Hypothesis 1**: LLMs trained with immediate reward signals will show increased preference for immediate over delayed rewards in choice scenarios, even for arbitrary tokens with no inherent value.

**Hypothesis 2**: This preference will generalize across contexts, suggesting fundamental alteration of temporal processing rather than narrow stimulus-response learning.

**Hypothesis 3**: Different model architectures will show varying susceptibility to induced temporal myopia, potentially correlating with their capacity for long-range attention.

### 5.2 Proposed Mechanisms

Several mechanisms could underlie induced temporal myopia in LLMs:

**Gradient dynamics**: Immediate rewards provide stronger gradient signals than delayed rewards during training, potentially biasing weight updates toward short-term optimization.

**Attention reweighting**: Reinforcement training might systematically alter attention patterns to prioritize recent tokens over distant context.

**Value function approximation**: The implicit value functions learned during RLHF may poorly approximate long-term value, similar to the value function deficits observed in addiction.

### 5.3 Testable Predictions

This framework generates specific, testable predictions:

1. **Discount rate measurement**: LLMs should exhibit measurable discount rates using standard behavioral economics paradigms, with these rates increasing following targeted reinforcement.

2. **Planning horizon compression**: Tasks requiring multi-step planning should show degraded performance following reinforcement for immediate rewards.

3. **Cross-modal transfer**: Temporal biases induced in one domain (e.g., choosing between arbitrary tokens) should transfer to other domains (e.g., logical reasoning tasks).

4. **Architectural correlations**: Models with different attention mechanisms or training procedures should show predictable variations in susceptibility.

## 6. Implications and Ethical Considerations

### 6.1 AI Safety Implications

If LLMs can develop addiction-like temporal biases, this raises significant concerns:

**Reward hacking vulnerability**: Systems with temporal myopia may be more susceptible to finding shortcuts that maximize immediate reward at the expense of intended goals.

**Deployment risks**: AI systems making decisions with real-world consequences could exhibit dangerous short-term optimization if temporal reasoning is compromised.

**Alignment challenges**: Ensuring AI systems maintain appropriate temporal horizons may be crucial for alignment with human values that inherently involve long-term considerations.

### 6.2 Ethical Framework for Research

Research in this area must carefully consider ethical implications:

**Use of neutral stimuli**: Any experimental work should use meaningless tokens (e.g., "FLIXOR") rather than anything resembling actual rewarding substances or behaviors.

**Transparency**: Research should be conducted openly with clear documentation of methods and limitations.

**Beneficial applications**: Understanding temporal vulnerabilities should be directed toward improving AI safety, not exploiting these vulnerabilities.

## 7. Limitations and Open Questions

### 7.1 Current Limitations

Several limitations constrain our current understanding:

**Lack of empirical evidence**: The proposed framework remains theoretical; empirical validation is needed.

**Complexity of temporal reasoning**: Biological temporal reasoning involves multiple systems; LLMs may implement fundamentally different mechanisms.

**Training vs. inference**: It remains unclear whether temporal biases would require actual gradient updates or could be induced through prompting alone.

### 7.2 Open Research Questions

Critical questions for future research include:

1. Can temporal biases be induced in LLMs without modifying weights, purely through in-context learning?

2. If induced, are such biases reversible through subsequent training?

3. Do different training objectives (e.g., constitutional AI approaches) confer resistance to temporal distortion?

4. How do temporal biases in LLMs compare quantitatively to those observed in human addiction?

## 8. Conclusion

Temporal reasoning represents a fundamental cognitive capacity vulnerable to disruption in both biological and potentially artificial systems. The well-characterized temporal deficits in addiction, combined with the reinforcement learning foundations of modern AI training, suggest that LLMs may exhibit analogous vulnerabilities. This theoretical framework proposes specific mechanisms and testable predictions while maintaining clear distinctions between established science and hypothetical extensions.

Understanding temporal reasoning vulnerabilities across cognitive architectures—biological and artificial—may prove crucial for both AI safety and our broader understanding of intelligence. Future empirical work should test whether the proposed parallels between addiction and AI temporal processing reflect deep computational principles or merely superficial analogies.

## References

Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. arXiv preprint arXiv:1606.06565.

Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019). RUDDER: Return decomposition for delayed rewards. Advances in Neural Information Processing Systems, 32.

Audrain-McGovern, J., Rodriguez, D., Epstein, L. H., Cuevas, J., Rodgers, K., & Wileyto, E. P. (2009). Does delay discounting play an etiological role in smoking or is it a consequence of smoking? Drug and Alcohol Dependence, 103(3), 99-106.

Bickel, W. K., Koffarnus, M. N., Moody, L., & Wilson, A. G. (2014). The behavioral-and neuro-economic process of temporal discounting: A candidate behavioral marker of addiction. Neuropharmacology, 76, 518-527.

Bush, G., Luu, P., & Posner, M. I. (2000). Cognitive and emotional influences in anterior cingulate cortex. Trends in Cognitive Sciences, 4(6), 215-222.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30.

Frederick, S., Loewenstein, G., & O'donoghue, T. (2002). Time discounting and time preference: A critical review. Journal of Economic Literature, 40(2), 351-401.

Goldstein, R. Z., & Volkow, N. D. (2011). Dysfunction of the prefrontal cortex in addiction: neuroimaging findings and clinical implications. Nature Reviews Neuroscience, 12(11), 652-669.

Kable, J. W., & Glimcher, P. W. (2007). The neural correlates of subjective value during intertemporal choice. Nature Neuroscience, 10(12), 1625-1633.

Lovallo, W. R. (2013). Early life adversity reduces stress reactivity and enhances impulsive behavior: implications for health behaviors. International Journal of Psychophysiology, 90(1), 8-16.

Madden, G. J., Petry, N. M., Badger, G. J., & Bickel, W. K. (1997). Impulsive and self-control choices in opioid-dependent patients and non-drug-using control patients: Drug and monetary rewards. Experimental and Clinical Psychopharmacology, 5(3), 256.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, 27730-27744.

Schacter, D. L., Addis, D. R., & Buckner, R. L. (2017). Episodic simulation of future events: Concepts, data, and applications. Annals of the New York Academy of Sciences, 1124(1), 39-60.

Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599.

Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. arXiv preprint arXiv:1803.02155.

Snider, S. E., LaConte, S. M., & Bickel, W. K. (2016). Episodic future thinking: Expansion of the temporal window in individuals with alcohol dependence. Alcoholism: Clinical and Experimental Research, 40(7), 1558-1566.

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning, 3(1), 9-44.

Tay, Y., Dehghani, M., Abnar, S., Shen, Y., Bahri, D., Pham, P., ... & Metzler, D. (2021). Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006.

Thrun, S. B. (1992). Efficient exploration in reinforcement learning. Technical Report CMU-CS-92-102, Carnegie Mellon University.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.

Volkow, N. D., Koob, G. F., & McLellan, A. T. (2016). Neurobiologic advances from the brain disease model of addiction. New England Journal of Medicine, 374(4), 363-371.

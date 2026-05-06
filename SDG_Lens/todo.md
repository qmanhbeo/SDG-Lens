1. Theoretical Bases & Technical Novelty
These points provide the academic "meat" for your Theory and Methods sections, situating your project within current AI research.

Transformer Scaling Laws: The project demonstrates the "classic transformer scaling pattern" where BERT-based models (MiniLM) eventually overtake simpler statistical baselines (TF-IDF + LinearSVC) as dataset size increases.  

Contextual Semantic Mapping: Unlike frequency-based models (TF-IDF), BERT's bidirectional architecture captures "contextual nuance," distinguishing between identical words used in different SDG contexts (e.g., "water" in SDG 6 vs. SDG 14).  

The "Attention as Explanation" Debate: The implementation uses attention weights as a proxy for feature importance, transitioning the model from a "Black Box" to a "Grey Box" system.  

Writing Tip: Cite the academic tension between Jain & Wallace ("Attention is not Explanation") and Wiegreffe & Pinter ("Attention is not not Explanation") to show deep theoretical engagement.

Transfer Learning Efficiency: The choice of a compact encoder (MiniLM) represents a strategic balance between computational efficiency and the linguistic richness required for complex multi-label classification.  

Multi-Label Synergy: The model learns shared linguistic features across overlapping goals (e.g., SDG 9 and 11), allowing it to handle the inherent semantic overlap of Sustainable Development Goals better than independent binary classifiers.  

Fine-Tuning Strategies: The decision to unfreeze only the last two transformer layers serves as a controlled "feature extraction" method, preventing catastrophic forgetting while adapting to SDG-specific vocabulary.  

2. Ethical & Societal Considerations
These points address the specific "Ethical and Societal Impacts" requirement of your assignment, framing your technical choices as moral ones.

Interpretability as an Ethical Mandate: Providing "Top Attended Tokens" is presented not just as a feature, but as an ethical choice to enable human-in-the-loop verification, reducing the risk of blindly following "black box" predictions.  

Automation Bias & High-Stakes Decision Making: The report must address the risk of policy-makers over-relying on automated labels for resource allocation, particularly where model "laggards" (like SDG 3 or 16) might lead to under-funding in critical areas.  

Data Representation & Bias: The reliance on an English-only subset and specific document-level labels (SDGi Corpus) introduces potential linguistic and cultural biases that could skew how SDGs are interpreted in non-Western contexts.  

Transparency in Error: By deliberately including "bad predictions" in the teammate brief, the project promotes "Honest AI" by mapping where the model fails and why (e.g., low scores or misleading attention tokens).  

Threshold Ethics: The choice of a lower classification threshold (0.3) reflects a prioritized "Recall" strategy—ensuring fewer relevant SDGs are missed, even at the cost of more "False Positives" that require human filtering.  

3. Discussion & Future Scaling
Use these points to bridge your Results and Conclusion sections.

Closing the Performance Gap: While TF-IDF currently wins on small data (2k examples), the "crossover point" at 4k suggests that BERT is the superior long-term solution for larger, more complex SDG datasets.  

Addressing Performance Variance: The uneven F1 scores across the 17 SDGs (e.g., strong performance in SDG 4 vs. weaker in SDG 3) highlight the need for class-balancing techniques like oversampling or data augmentation in future iterations.  

User Trust and Adoption: The "SDG Lens" serves as a prototype for building trust with non-technical stakeholders (NGOs, UN researchers) by providing inspectable evidence for every prediction.
# Reproducing HateXplain for Explainable Hate Speech Detection

## Introduction

This report reproduces **HateXplain**, a socially relevant AI study on explainable hate-speech detection. The original research asks whether hate-speech models can be both accurate and interpretable, rather than acting as black boxes that affect online moderation without clear reasons. This matters because content moderation systems influence participation, visibility, and safety in everyday social life. If such systems wrongly flag minority users or fail to identify harmful speech, they can distort public discussion and reinforce social inequality. HateXplain is especially relevant because it studies three connected problems at the same time: classification, explanation, and bias. In my reproduction, I found that this combined design is the paper’s main strength, but also the main source of difficulty. Reproducing classification scores is manageable. Reproducing explanation and subgroup results is much more sensitive to tokenisation, rationale aggregation, and evaluation choices.

## Original paper summary

HateXplain introduces a benchmark dataset for explainable hate-speech detection. It contains **20,148** posts from **Twitter and Gab**. Each item is annotated by three crowd workers. The annotations include: a 3-class label (**hatespeech**, **offensive**, or **normal**), a target-community label, and token-level rationales showing which words supported the annotator’s decision. The paper excludes “undecided” cases where all three annotators disagree, and then evaluates several neural models, including CNN-GRU, BiRNN, BiRNN with attention, and **BERT**. The most relevant baseline for my reproduction is **BERT [Attn]**, which reports **accuracy 0.690**, **macro-F1 0.674**, and **AUROC 0.843**. The paper also shows that stronger classification does not automatically mean better explanations. Its explainability analysis uses plausibility metrics such as **Token-F1** and **IOU-F1**, and faithfulness metrics such as **comprehensiveness** and **sufficiency**. It further argues that rationale-supervised models can reduce unintended bias toward target communities.

## My reproduction summary

I reproduced the paper with a clean, assessor-friendly pipeline that can run once from a single command. Instead of submitting unrelated baseline code, I built a package specifically around the same research object as the paper: the public HateXplain data, the official `post_id_divisions.json` split file, and a **BERT-based classifier with attention supervision from human rationales**.

The code automatically downloads `dataset.json` and `post_id_divisions.json` from the public repository, verifies file checksums, and constructs deterministic train, validation, and test splits by `post_id`. It then tokenises the posts with `bert-base-uncased`, keeps alignment between original tokens and WordPiece tokens, and builds a rationale target vector for each example using the majority rationale across annotators. I excluded undecided label cases to remain consistent with the paper’s reported setup.

For modelling, I used a BERT encoder with a linear classification head. The training loss combines standard cross-entropy for 3-class prediction with an auxiliary attention-supervision loss. Concretely, the model uses the final-layer `[CLS]` attention distribution and encourages it to place weight on tokens marked by human rationales. This is not a copy of the original research code. It is a clean reimplementation of the same modelling idea, designed to be more reproducible in coursework conditions.

The evaluation module reports classification metrics, plausibility metrics, faithfulness checks, and a simple subgroup bias audit by target community. Results are exported to `outputs/results.json`. I also created **fast** and **full** configurations. The fast version is CPU-friendly for marking. The full version uses the complete split and is closer to the original experimental setting.

## Critical evaluation of the research design

The paper’s strongest contribution is its multi-dimensional design. Many hate-speech studies only ask whether a classifier is accurate. HateXplain asks a broader and more important question: **accurate for whom, and on what grounds?** This is valuable because real moderation systems affect speech rights, platform safety, and trust. The dataset’s three-layer annotation structure is therefore a genuine methodological advance.

At the same time, the study faces several design problems. First, the boundary between **hate speech** and **offensive speech** is socially and contextually unstable. Even with three annotators, disagreement remains substantial. That means small performance differences between models may partly reflect label ambiguity rather than true model superiority. Second, rationale labels are not objective facts. They are human interpretations of why a post should be classified in a certain way. This makes explanation metrics useful, but also fragile. In my reproduction, plausibility scores were clearly more sensitive than classification scores because they depend on exact token boundaries, majority-vote aggregation, and the way top-k rationale tokens are selected.

Third, bias evaluation is necessary but difficult. The paper is right to emphasise target communities, because moderation errors can burden already marginalised groups. However, subgroup auditing depends on the number of examples for each target category and on the exact bias metric used. A model may look fair under one metric and less fair under another. For that reason, I used a transparent subgroup audit in the reproduction package and reported community-level support alongside error measures. This does not eliminate bias, but it makes the evaluation easier to inspect.

From a technical perspective, the nature of the design is **experimental and benchmark-based**. It is not causal research and does not show how moderation systems affect real users after deployment. Its value lies in controlled comparison under a shared dataset. My reproduction confirms that this design is useful for comparing model families, but limited for making broad social claims. Another weakness is reproducibility. The original repository is a research codebase, not a grading-friendly software package. I therefore changed the implementation strategy: I kept the official public data and official splits, but reimplemented the training and evaluation pipeline in a deterministic, one-command form. This was a necessary technical modification, not a change in the research question.

Overall, HateXplain remains a strong benchmark for socially relevant AI. Its main insight still holds: high classification performance alone is not enough. A useful moderation model should also provide plausible rationales and be audited for uneven errors across target communities. My reproduction supports that conclusion while also showing that explanation-oriented NLP benchmarks require much stricter packaging and logging than ordinary accuracy-focused tasks.

## References

Mathew, B., Saha, P., Yimam, S.M., Biemann, C., Goyal, P. and Mukherjee, A. (2021) ‘HateXplain: A benchmark dataset for explainable hate speech detection’, *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(17), pp. 14867–14875.

Hate-ALERT (2026) *HateXplain repository*. Available at: https://github.com/hate-alert/HateXplain

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) ‘BERT: Pre-training of deep bidirectional transformers for language understanding’, *NAACL-HLT*, pp. 4171–4186.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D. and Gebru, T. (2019) ‘Model cards for model reporting’, *FAT*.

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d’Alché-Buc, F., Fox, E. and Larochelle, H. (2021) ‘Improving reproducibility in machine learning research’, *Journal of Machine Learning Research*, 22(164), pp. 1–20.

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J.W., Wallach, H., Daumé III, H. and Crawford, K. (2021) ‘Datasheets for datasets’, *Communications of the ACM*, 64(12), pp. 86–92.

Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017) ‘Automated hate speech detection and the problem of offensive language’, *Proceedings of ICWSM*.

Sap, M., Card, D., Gabriel, S., Choi, Y. and Smith, N.A. (2019) ‘The risk of racial bias in hate speech detection’, *Proceedings of ACL*, pp. 1668–1678.

# Model card

## Model name
HateXplain aligned reproduction

## Intended use
This model is for coursework reproduction of the HateXplain benchmark. It is intended for research and demonstration only.

## Not intended for deployment
This package must not be used as a production moderation system. Hate-speech classification is socially sensitive and can create harmful false positives and false negatives.

## Data
The model uses the public HateXplain dataset and the official post ID split file released by the authors.

## Output classes
- hatespeech
- offensive
- normal

## Explanation mechanism
The package supervises final-layer `[CLS]` attention with human rationale masks and then evaluates word-level rationale overlap.

## Main limitations
- label boundaries are socially contingent
- rationale annotations are not objective ground truth
- subgroup bias depends on target-community sample size
- benchmark results do not equal real-world moderation performance

## Responsible-use note
Repository users should keep offensive content warnings visible and should never describe benchmark scores as evidence of fairness in real deployment.

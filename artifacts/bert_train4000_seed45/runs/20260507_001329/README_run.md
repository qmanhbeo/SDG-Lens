# SDGi Lens Run 20260507_001329

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint /home/manh/study-BHam/AI4GC-2/artifacts/bert_train4000_seed45/runs/20260507_001329/model.pt
```

## Metrics

- micro-F1: 0.7142411966344656
- macro-F1: 0.7031483727362656
- weighted-F1: 0.7130175259539419
- subset accuracy: 0.651844843897824
- average predicted labels: 1.5950804162724692

## Config

- model: sentence-transformers/all-MiniLM-L6-v2
- data: /home/manh/study-BHam/AI4GC-2/data
- train/test rows: 4000 / 1057
- epochs: 3
- threshold: 0.3
- encoder mode: last_2_layers
- command: `python /home/manh/study-BHam/AI4GC-2/scripts/train.py --seeds 42 43 44 45 46 --train-sizes 1000 2000 4000 --language en --test-seed 43 --device cuda --force`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.

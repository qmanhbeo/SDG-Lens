# SDGi Lens Run 20260506_234818

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint /home/manh/study-BHam/AI4GC-2/artifacts/bert_train1000_seed46/runs/20260506_234818/model.pt
```

## Metrics

- micro-F1: 0.6297502714440825
- macro-F1: 0.6081964399487797
- weighted-F1: 0.6236285664873689
- subset accuracy: 0.6166666666666667
- average predicted labels: 1.44

## Config

- model: sentence-transformers/all-MiniLM-L6-v2
- data: /home/manh/study-BHam/AI4GC-2/data
- train/test rows: 1000 / 300
- epochs: 3
- threshold: 0.3
- encoder mode: last_2_layers
- command: `python /home/manh/study-BHam/AI4GC-2/scripts/train.py --seeds 42 43 44 45 46 --train-sizes 1000 2000 4000 --language en --test-seed 43 --device cuda --force`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.

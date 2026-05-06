# SDGi Lens Run 20260506_232618

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint /home/manh/study-BHam/AI4GC-2/artifacts/bert_train4000_seed43/runs/20260506_232618/model.pt
```

## Metrics

- micro-F1: 0.7035897435897436
- macro-F1: 0.679585814731445
- weighted-F1: 0.7002512538755662
- subset accuracy: 0.68
- average predicted labels: 1.62

## Config

- model: sentence-transformers/all-MiniLM-L6-v2
- data: /home/manh/study-BHam/AI4GC-2/data
- train/test rows: 4000 / 300
- epochs: 3
- threshold: 0.3
- encoder mode: last_2_layers
- command: `python /home/manh/study-BHam/AI4GC-2/scripts/train.py --seeds 42 43 44 --train-sizes 2000 4000 --language en --test-samples 300 --test-seed 43 --device cuda --force`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.

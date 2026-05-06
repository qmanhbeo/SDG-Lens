# SDGi Lens Run 20260506_232530

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint /home/manh/study-BHam/AI4GC-2/artifacts/bert_train4000_seed42/runs/20260506_232530/model.pt
```

## Metrics

- micro-F1: 0.6991701244813278
- macro-F1: 0.6760656321993372
- weighted-F1: 0.6925385219303791
- subset accuracy: 0.6633333333333333
- average predicted labels: 1.5833333333333333

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

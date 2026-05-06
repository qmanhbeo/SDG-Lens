# SDGi Lens Run 20260506_234805

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint /home/manh/study-BHam/AI4GC-2/artifacts/bert_train1000_seed45/runs/20260506_234805/model.pt
```

## Metrics

- micro-F1: 0.610917537746806
- macro-F1: 0.5952383892163565
- weighted-F1: 0.6036409425411539
- subset accuracy: 0.6066666666666667
- average predicted labels: 1.24

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

# SDGi Lens Run 20260503_150722

This run train_eval a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python run.py --load outputs/runs/20260503_150722/model.pt --eval-only
```

## Metrics

- micro-F1: 0.7049180327868853
- macro-F1: 0.6868580479997828
- weighted-F1: 0.7034547656619351
- subset accuracy: 0.6666666666666666
- average predicted labels: 1.6233333333333333

## Config

- model: sentence-transformers/all-MiniLM-L6-v2
- data: /home/manh/study-BHam/AI4GC-2/SDGi/data
- train/test rows: 4000 / 300
- epochs: 3
- threshold: 0.3
- encoder mode: last_2_layers
- command: `python run.py --train_samples 4000 --test_samples 300 --epochs 3 --output_dir outputs/runs/20260503_bert_4k`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.

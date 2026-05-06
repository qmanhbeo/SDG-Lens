# SDGi Lens Run 20260503_134910

This run eval_only a minimal SDGi multi-label classifier and saved an
attention-based explanation proxy. Attention is not ground-truth rationale.

## Reload

```bash
python main.py eval --checkpoint outputs/runs/20260503_134910/model.pt
```

## Metrics

- micro-F1: 0.6614699331848553
- macro-F1: 0.6424677473768673
- weighted-F1: 0.6574711456731414
- subset accuracy: 0.6566666666666666
- average predicted labels: 1.3633333333333333

## Config

- model: sentence-transformers/all-MiniLM-L6-v2
- data: data
- train/test rows: 2000 / 300
- epochs: 3
- threshold: 0.3
- encoder mode: last_2_layers
- command: `python main.py eval --checkpoint outputs/runs/20260503_134841/model.pt`

## Limitations

Prototype only: no scheduler, no checkpoint resume, no class weighting, and no
SDGi token-level rationale labels for explanation validation.

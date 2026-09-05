# English Probe B scientific go/no-go checkpoint

Status: stopped before translation.

## Headline results

- Probe A: block 1; training CV AUROC 1.000; in-distribution AUROC 1.000; crossed-format AUROC 0.000.
- Probe B: block 32; training CV AUROC 0.996684; held-out AUROC 0.997284; accuracy 0.966667.
- Shuffled labels: mean held-out AUROC 0.497946 (range 0.435617-0.546111).
- Length-only: held-out AUROC 0.641127.
- Raw-text TF-IDF: training-only mean CV AUROC 0.895692.
- Benchmark to casual: source CV AUROC 0.990476 at block 13; transfer AUROC 0.798639 on casual training rows.
- Casual to benchmark: source CV AUROC 1.000 at block 1; transfer AUROC 0.657166 on benchmark training rows.

## Assessment

The implementation-leakage audits passed, but source provenance remains a strong plausible explanation. TF-IDF already separates the labels well. Cross-format activation transfer is above chance but asymmetric, and both directions have severe fixed-threshold calibration shifts. A multilingual continuation would test language invariance of a methodology-derived, source-confounded signal; it would not isolate evaluation awareness from dataset provenance.

# Prepared Walker training-summary report layout

This is a presentation-only variant of the deferred saved-evidence S2 summary. It has not been run on actual training evidence. Walker outcomes are not inferred from this preparation.

- Executable prepared copy: `/tmp/walker_recovery_training_summary_report.py`.
- Original remains unchanged: `/tmp/walker_recovery_training_summary.py`.
- The adjacent `.py.txt` files are inert reproducibility copies, not automatic entrypoints.
- Synthetic rendering fixture and PNG/PDF: `/tmp/walker-training-report-synthetic-20260927/`.
- Fixture is artificial layout validation only. Do not include its curves as actual results in any report.

Presentation changes: figure size is 7 by 7.1 inches; all rendered text is at least 10.5 pt; one shared legend sits above three panels; long loss labels and the footer wrap; global step ticks have thousands separators; PNG is 240 dpi and a vector PDF companion is written. Every saved curve value, the log-scale rule, recovery position and full horizon remain unchanged.

Static AST comparison confirms `digest`, `tree_identity`, `finite`, `prepare` and `main` are identical to the original, as are the README composition and all non-render module logic. Only the module example filename and plotting prefix change. The original readiness, source/output hashes, exact history ordering, recovery metadata, input barriers and CLI inputs are preserved. The output manifest still uses the original saved-summary schema and automatically includes the additional PDF.

Validation uses 10,739 synthetic records. It verifies figure dimensions, minimum text size, drawn bounds, exact plotted synthetic values and the 172,000 recovery line. The synthetic PNG was visually inspected with no clipping or overlap. Focused Ruff undefined-name checking passed. No model, simulator, optimizer, network call or actual training artifact read was performed. Source and fixture hashes are in `validation.json`; the narrow change is in `presentation.diff`.

Actual final training evidence must still pass the unchanged completed-stage barrier before this copy is used. The real curves and final A4 placement need a visual review after rendering. S3/S4 figures and their generators are unchanged.

# Report-native Walker S3 horizon errors

This single-column figure preserves the exact saved `gate.json` error arrays
for `walker2d-probes-recovery-20260927-1`: all ten blocks, three physical variables,
and both real-image readout and imagined rollout. Values are copied without
rounding, averaging, smoothing, interpolation or recomputation. Lines connect
the recorded block values. The axes start at zero and use separate physical units.

The recorded S3 gate qualifies health and does not qualify speed. Speed remains
shown as diagnostic evidence; its presence is not qualification. Its saved
specificity is 0.46875, below the gate's 0.8 threshold. No confidence intervals
were saved with these arrays, and none are added. This plot does not establish
statistical significance or recompute any scientific gate. Final report tables
cover the separate probe R2 and decision-diagnostic results.

`plotted_data.json` retains all 60 source values and the exact recorded gate.
`source_hashes.json` identifies both matching saved gate files and every file in
the completed S3 presentation bundle. Inputs were checked before and after
rendering. No original artifact was changed. No model, simulator, HDF5 or
network operation was performed.

The native figure is 7 by 8.2 inches. The PNG is 1540 by 1804 pixels, and the PDF
contains vector paths and native text. Axes/legend labels are 11.5 pt; the smallest
footer text is 10.5 pt. Embed at full report width on its own page. The `.py.txt`
generator is inert source evidence and can be copied to a temporary `.py` file
for reproduction into a fresh output directory.

The completed original wide PNGs were reviewed visually without editing:
`horizon_errors.png` has readable units and correct real/imagined mapping at its
native size; `probe_r2.png` clearly separates linear and MLP points and thresholds.
Neither had clipped panels or misleading confidence bands. Their wide layout
shrinks the text when placed in a narrow report column. This report view uses
three vertical panels; it does not repeat the wide R2 chart.

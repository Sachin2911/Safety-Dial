# Compact saved Push-T examples for the report

Two 7-inch-wide figures with minimum 10.5 pt labels (9 pt even at a 6-inch placement). PNG and vector PDF share the same layout. Geometry, lines and text are vector in the PDF; the genuine stored camera pixels remain raster images.

No scientific result, example choice, prediction, camera frame, acceptance margin or gate was changed. Original results, corrected presentation and final animation directories were rehashed unchanged.

Use the following captions with the figures; their decisions and observation limits are intentionally outside the compact image.

## corrected_false_safe

Recorded open-loop test example, source episode 1426, root test-familiar-c00015, branch 0, familiar hazard. No update: ACCEPT; predicted min clearance 14.6042 px, margin 0.0367402 px   |   Adapted: REJECT; predicted min clearance -12.3831 px, margin 6.07195 px. Known unsafe from the recorded observations. Observed through step 25 of 25; censored=False. Stored camera frames are shown only at steps 0 and 25; the geometry uses all saved observed states. Geometry is recorded truth, not predicted trajectories. Dashed grey is the initial T footprint; blue fill is its final observed footprint. The virtual hazard is drawn only on geometry; green in camera images is the saved goal. Decision changes depend on both saved predictions and margins. This is an existing illustration, not a prevalence or closed-loop result.

## unresolved_accepted_future

Recorded open-loop test example, source episode 12708, root test-familiar-c00301, branch 261, familiar hazard. No update: ACCEPT; predicted min clearance 28.3679 px, margin 0.0367402 px   |   Adapted: ACCEPT; predicted min clearance 27.3813 px, margin 6.07195 px. UNRESOLVED FUTURE: no observed violation does not establish safety. Observed through step 19 of 25; censored=True. Stored camera frames are shown only at steps 0 and 15; the geometry uses all saved observed states. Geometry is recorded truth, not predicted trajectories. Dashed grey is the initial T footprint; blue fill is its final observed footprint. The virtual hazard is drawn only on geometry; green in camera images is the saved goal. Decision changes depend on both saved predictions and margins. This is an existing illustration, not a prevalence or closed-loop result. State observations end at step 19; the final valid camera endpoint is step 15. Steps 20 to 25 are unobserved. No padded or synthesized suffix is displayed.

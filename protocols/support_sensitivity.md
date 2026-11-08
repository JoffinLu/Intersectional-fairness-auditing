# Prespecified supplemental real-support sensitivity

This protocol is written before generating any supplemental errors. It addresses
the limited support and target-size coverage of the original application
experiment. It does not change the original benchmark or establish overall
methodological novelty.

- Freeze exactly 32 configurations in this order: Adult, Diabetes, ACS Income,
  ACS Employment; within task, LR then HGB; within model, capped then uncapped
  counts; within count geometry, smallest eligible then smallest count at least
  80. All configurations and their complete probability/count vectors must be
  saved before the first simulation call. No cell, seed, or configuration may
  subsequently be chosen for favorable outcomes.
- Use the corrected processed caches. The selected family is identical across
  count geometries: audit positives at least 10 and calibration positives at
  least 5. Retain original group order. First round the original audit-positive
  counts down to multiples of 10. The capped geometry additionally takes the
  minimum with 1,000; the uncapped geometry starts from the original counts and
  applies no cap. It must not be reconstructed from capped values.
- The first target is the first group attaining the smallest transformed count.
  The second is the first group attaining the smallest transformed count among
  groups whose transformed count is at least 80. Abort preparation if either
  rule is undefined; do not substitute a different target. Ties follow original
  cache group order. Every method in a configuration uses that same target and
  the same total group counts.
- Use only the calibration-heterogeneous baseline: for each frozen predictor,
  `(calibration false negatives + .5)/(calibration positives + 1)`, clipped to
  `[.02,.8]`. Inject the designated target's error probability as the
  count-weighted complement plus .25, capped at .95. Recompute every group's
  true complement disparity from the final probability vector and the selected
  count geometry. Do not assume that only the injected target is harmful.
- Generate independent Bernoulli errors conditional on these fixed templates.
  Their truth is known only for this generated model, not for the original
  populations or observed errors. Full-data and split methods have the same
  total data budget; the exact .8 confirmation fraction preserves complement
  weights because all counts are multiples of 10.
- Reuse `run_simulation.evaluate_counts` without editing it: confirmation .8,
  weight floor .1, alpha .05, tolerance .05, and its existing HB nuisance budget
  `eta = alpha*.05/(20*groups)`. Retain full HB, full Hoeffding and all five
  split Holm methods. Basic point diagnostics returned by that function are
  retained, but no additional SR/SureMap subset is run.
- Fix seed 20261024 and 1,000 repetitions for each of the 32 configurations.
  Use NumPy SeedSequence child streams in the frozen configuration order.
  Comparisons within each configuration reuse its generated discovery and
  confirmation counts; cross-configuration differences are not called paired.
- Save raw NPZ counts, probabilities, truths, estimates, decisions and weights;
  complete configurations; all eligible and excluded support counts; source,
  protocol and processed-data SHA256 hashes; actual repetitions and wall time.
  Summaries report actual target flagging and familywise false-flag frequencies,
  Monte Carlo standard errors, and pointwise 95% Wilson intervals. For every
  other calibrated method, report target-power and FWER differences from full
  HB using within-repetition differences and paired MCSEs. Zero-event MCSEs
  remain zero, with Wilson upper bounds retained. Screen false-alert frequencies
  are diagnostics, not calibrated FWER guarantees.
- Preparation and execution are separate commands. Preparation performs no
  random outcome generation. Execution verifies the frozen protocol/code/data
  hashes and refuses an already started or completed run. Execution must follow the frozen protocol without editing source or inputs.

The output is `results/support_sensitivity_v1`. Summarize all 32 scenarios,
including null, unfavorable, and duplicate-outcome cases if any. Interpret
maximum differences and counts of wins descriptively, without claiming
simultaneous inference or selecting a new main baseline.

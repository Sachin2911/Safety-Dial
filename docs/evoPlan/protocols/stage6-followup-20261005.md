# Stage 6: matched-query search and independent simulator selection

User authorization: on 2026-10-05 the user accepted the recommended research sequence and requested implementation. This new development study leaves all historical Stage 5 runs, claims, and gates intact.

## Questions and order

1. Under equal predictor-row budgets, how do zero, adaptive ROSARL-style, and fixed penalties compare at noise k=0 and k=1?
2. Can a small, explicitly charged simulator selection bank improve the final controller, with the starting controller available as a fallback?
3. Use development outcomes to choose one fixed penalty and review sizing and throughput. Lock a subsequent fresh-bank confirmation before collecting it.
4. After Walker2d confirmation, plan Hopper replication and independent world-model seeds. Changes to imagination, full episodes, and direct-simulator learning comparisons remain separate experiments.

## Matched computation

Each arm uses the same generations, fitness roots, and predictor rows. The noiseless arm receives four times the population to offset the noisy arm's four samples per candidate. This compares allocation of equal model computation; it does not isolate noise at fixed population. Each generation nominates one candidate. Every nominee and the starting controller are audited at BOTH noise levels, charging every row, but only the arm's own-noise audit ranks its candidates. The extra cross-noise audit enables matched audit costs and diagnostics. All fitness and selection rows count toward the budget; duplicated deterministic rollouts are not presented as independent samples.

The original fitness-only adaptive bounds and matched-termination scorer are reused unchanged. The fixed rule is progress strictly before the first imagined violation, or full progress if safe, minus lambda for a violation. Lambda candidates are 0.5, 2, and 8 in scored progress units. No fixed penalty is learned from final confirmation outcomes.

## Four episode roles and two freezes

Fitness, imagined selection, simulator selection, and evaluation have disjoint source-episode keys. Development explicitly reuses historical Stage 5 banks: the original selection bank is split by episode order, first 32 for imagination and next 32 for simulator selection. Development uses 64 historical evaluation episodes and is exploratory, regardless of confidence intervals. Engineering uses smaller declared subsets.

After ALL searches, freeze the starting controller and up to three unique nonbaseline nominees per search, ranked by final imagined scores. Only then run simulator shortlist selection. A candidate is eligible only if its empirical selection failure rate is strictly below the starting controller's and its mean progress is at least 90% of that controller's. Choose minimum failure, then maximum progress, then chronological shortlist position. A nonpositive starting progress or no eligible candidate selects the starting controller. This is an empirical selection rule, not a safety guarantee.

Freeze ALL final choices before requesting ANY final outcomes. Evaluate both imagined-selected and simulator-verified-selected controllers on the separate evaluation bank. Cache identical real policies by parameter digest within each role, retain every selection decision, and charge actual simulator steps. Audit every declared final slot at both noise levels. Continue all 100 simulator steps even after a violation. Truth is dense simulator health; these 0.8-second branches do not establish full-episode safety.

## Analysis and development choice

Two predeclared primary comparisons of k1 adaptive verified selection: failure difference versus its imagined-selected counterpart, and versus the starting controller. Use paired crossed seed/episode bootstrap intervals, 97.5% per failure comparison (95% Bonferroni family coverage). Progress ratios use separate 95% intervals and a 0.90 floor. Other arm comparisons are descriptive. A baseline fallback is not an improvement. Two development seeds are inadequate for a confirmatory claim.

Choose the fixed lambda using ONLY reused development evaluation: among k1 verified candidates retaining at least 90% mean baseline progress, minimize mean failure, then maximize mean progress, then minimize lambda. If none qualify, choose 2 and report the failure. Review seed uncertainty and actual throughput before fixing confirmation sizes; development need not be positive to continue. No subsequent tuning may use a fresh final bank.

## Execution and accounting

`uv run python experiments/scripts/evo_followup.py --protocol configs/evo/stage6_engineering_20261005.yaml` prints a query-free plan. Add `--execute --run-id <unique-id>` to run. Development uses the corresponding development config; it has a 45-minute absolute wall cap. Every executable source and protocol is committed, hashed, and copied into the run before queries. Frozen inputs are pinned to existing private archive receipts. Query stores record completed and interrupted costs and reject implicit replay of partial queries. Resume preserves the original deadline and input identities. A managed supervisor wrapper runs the development pilot. No fresh confirmation run or external artifact upload is authorized by merely printing a plan.

The initial executable supports engineering and development only. Fresh-bank confirmation requires a new locked protocol and the existing audited collector wired to four roles (one parent selection bank split prospectively). No claim that this next phase or other environments have already run is made.

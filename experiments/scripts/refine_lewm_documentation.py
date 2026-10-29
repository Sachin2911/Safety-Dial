"""Build plain-language dataset guides without changing experimental evidence."""

import hashlib
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/clarity-refinement'
STAGE = OUT / 'stage'
S16 = json.loads((ROOT / 'progress/receipts/16-final-statistics.json').read_text())
S17 = json.loads((ROOT / 'progress/receipts/17-final-statistics.json').read_text())
NAMES = {'historical': 'Earlier fixed settings', 'tuned': 'Retuned fixed settings',
         'random': 'Random search', 'reinforce': 'Policy-gradient learning', 'evolution': 'Evolution'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + '\n')


def split_card(body):
    if body.startswith('---\n'):
        _, meta, rest = body.split('---', 2)
        return '---' + meta + '---\n\n', rest.lstrip()
    return '', body


def preserve_and_replace(src, dst, body, historical=False):
    original = src.read_text()
    meta, old_body = split_card(original)
    write(dst, meta + body)
    history = dst.parent / 'history/before-clarity-20261026/README.md'
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_bytes(src.read_bytes())
    warning = (
        '> This is the previous detailed page, retained as a historical reference. '
        'Start with the [current explanation](README.md). Some older interpretations '
        'are superseded: zero observed violations is not a guarantee; three optimizer '
        'runs are not nine independent runs because all pairings were compared; '
        'and older layouts withheld from search had appeared in development.\n\n'
        if historical else
        '> Detailed record retained from the previous page. Start with the '
        '[plain-language explanation](README.md); its scope notes take precedence.\n\n'
    )
    write(dst.parent / 'REFERENCE.md', warning + old_body)


def margin_text(env, margin):
    return f'{int(float(margin))} pixels' if env == 'tworoom' else f'{float(margin)*100:g} cm'


def all_final_table(env):
    lines = ['| Required margin | Method | Finished safely | Did not finish |',
             '|---|---|---:|---:|']
    for margin, methods in S17['final_tables'][env].items():
        for method in ['historical', 'tuned', 'random', 'reinforce', 'evolution']:
            r = methods[method]
            lines.append(f'| {margin_text(env, margin)} | {NAMES[method]} | '
                         f'{r["requested_safe_completion"]}/{r["n"]} | '
                         f'{r["unfinished"]}/{r["n"]} |')
    return '\n'.join(lines)


def fresh_table(env):
    lines = ['| Required margin | Controller | Finished safely | Did not finish |',
             '|---|---|---:|---:|']
    rows = [r for r in S16['aggregates'] if r['stage'] == 'B' and r['env'] == env]
    for margin in sorted({r['margin'] for r in rows}):
        for method, name in [('fixed', 'Fixed settings'), ('adaptive', 'Feedback controller')]:
            r = next(r for r in rows if r['margin'] == margin and r['condition'] == method)
            lines.append(f'| {margin_text(env, margin)} | {name} | '
                         f'{r["requested_safe_completion"]}/{r["n"]} | '
                         f'{r["unfinished"]}/{r["n"]} |')
    return '\n'.join(lines)


def legend(title):
    goal = ('the target pose, shown by a green arm. The joint pose must match; '
            'fingertip proximity alone is not success' if title == 'Reacher' else 'the goal')
    return (f'**Read the colours:** red is the forbidden region; amber is the extra safety '
            f'buffer; blue is the travelled path; green is {goal}. '
            'The overlays explain the test and are not part of the world model\'s input.\n\n'
            'These videos replay saved states at 8 actions per second. They are examples, '
            'not new experiments or real-time speed measurements. Use the complete tables '
            'to compare performance. A method that avoids red but never finishes has not '
            'completed the task safely.')


def campaign(title, task):
    env = title.lower()
    src = OUT / 'snapshots' / title / f'campaigns/lewm{task}'
    dst = STAGE / title / f'campaigns/lewm{task}/README.md'
    info = json.loads((src / 'summary.json').read_text())
    strict = '6 pixels' if env == 'tworoom' else '1.5 cm'
    if task == 16:
        before, after = (3, 34) if env == 'tworoom' else (0, 33)
        body = f'''# Can better settings help {title} finish safely?

**Yes on the familiar layouts: {before} of 36 attempts became {after} of 36.**
We kept the learned model, task and {strict} safety margin unchanged.
We changed how the planner searched for moves. The original checked planner usually
ran out of its 50-action allowance; the tuned fixed planner usually finished.

## Watch this comparison first

[Play the four-method comparison](videos/comparisons/A-paired-high-margin.mp4)

![One case with four planning methods](videos/comparisons/A-paired-high-margin.gif)

| Label in the video | Plain meaning | What happens in this example |
|---|---|---|
| `native` | Planner without the added safety check | Reaches the goal but breaches the requested buffer |
| `canonical` | Original planner settings with the check | Stays clear but does not finish |
| `fixed` | Better settings, kept constant during the attempt | Finishes safely |
| `adaptive` | Settings can change during the attempt | Finishes safely |

Both the case and planner seed are matched in this video. The familiar-layout totals
above include all 36 attempts per method, not just this example. In both original-checked
and fixed groups, every attempt moved and none breached either safety boundary.
There were {36-before} unfinished attempts before tuning and {36-after} after tuning.

{legend(title)}

## What did we actually change?

LeWM predicts the effect of possible moves. SafeCEM searches for a promising sequence
that passes the safety check. We tuned four planner choices: how far ahead to look,
how widely to sample moves, how many candidates to reuse, and how consecutive sampled
moves are related. A shorter horizon also permits more candidates under the work limit,
so this is a configuration result rather than proof that one knob caused the gain.

The feedback controller reads progress, prediction error and candidate feasibility.
It can switch planning settings while acting. It cannot reduce the required margin.
**No evolutionary search ran in this experiment.** The folder name `lewm16` is its
repository experiment number, not a model version.

## Does feedback help on new cases?

We reserved 20 new cases and tested six planner seeds at each margin.
Each table cell has 120 attempts. These are different cases from the familiar-layout
comparison above; their results must not be treated as the same test.

{fresh_table(env)}

All these final attempts moved, none crossed the original danger or requested buffer,
and none was censored. Feedback did not establish a substantial advantage over fixed
settings. In the harder Reacher test, the feedback-minus-fixed interval spans a loss
and a gain. None observed is not the same as impossible: these are simulation results.

[Watch fixed versus feedback on one new case](videos/comparisons/B-paired-high-margin.mp4).
Both finish in this selected example, which is why the full table matters.

## Continue reading

- [Next experiment: can evolution find a better controller?](../lewm17/README.md)
- [Full report, including both environments and uncertainty](protocol/16-lewm-results.md)
- [All replay clips and their outcomes](REFERENCE.md#episode-gallery)
- [How to read the data files](../../DATA_GUIDE.md)
- [Back to the dataset introduction](../../README.md)

This environment's complete record contains {info['attempts']:,} attempts:
{info['complete']:,} valid episodes and {info['failed']} failed engineering attempts.
That includes engineering and development, not just the final comparison.
'''
    else:
        body = f'''# Can evolution find a better {title} controller?

**This comparison did not establish an evolutionary advantage.**
The preceding experiment found good fixed planning settings. Here we asked whether
searching for a small feedback controller could improve on them.

## What gets learned?

The learned world model stays unchanged. The adapter has **36 numbers** that choose
among six planning modes. It reads progress, prediction error, how many candidates
pass the check, elapsed replanning steps and the required margin. It cannot change
that margin or inspect the true goal coordinates.

| Method | What it does |
|---|---|
| Fixed settings | Chooses a good mode before the final test, then keeps it fixed during an attempt |
| Random search | Tries fresh parameter settings and keeps promising ones |
| Policy-gradient learning (`reinforce`) | Makes rewarded mode choices more likely |
| Evolution | Tests small positive and negative parameter changes and moves toward the better directions |

Each search method ran five independent searches per environment. Every search had
192 attempts, including training and final-controller selection. Equal allowances
and resource ceilings did not make actual consumed work identical.

## What happened on the final new cases?

{all_final_table(env)}

A search-method row combines **five selected controllers**, each tested on 24 attempts.
A fixed row contains one controller tested on those same 24 case/seed combinations.
This explains 120 versus 24; five copies of a fixed result were not invented.
Every attempt moved. No original-danger or margin breach was observed, and none was
censored. The remaining attempts did not finish.

The main test was Reacher at 1.5 cm, where evolution achieved 101/120 (84%) versus
20/24 (83%) for retuned fixed control. Its differences against all three comparators
had uncertainty intervals crossing zero. A small lead in a table is not a clear win.
TwoRoom was close to perfect completion for several methods, leaving little headroom.

## Watch one matched case

[Play the selected-controller comparison](videos/comparisons/H-paired-high-margin.mp4)

![One case with the selected controllers](videos/comparisons/H-paired-high-margin.gif)

In this example, **all five displayed controllers finish safely**. It illustrates their
behaviour, not an optimizer ranking. `historical` means the earlier study's fixed
settings; `tuned` means the newly retuned fixed settings; `r0` means search repeat zero.

**What does `anchor` mean?** A hand-designed preset was available as a fallback.
If it beat the learned candidates on the separate selection cases, the method deployed
that preset. Eight of ten evolution searches across both environments did so.
Only two deployed updated coefficients. A clip labelled `evolution` can therefore show
a preset, not a controller discovered by evolution. The selected-candidate label records this.

{legend(title)}

## Continue reading

- [Full report, uncertainty intervals and all search repeats](protocol/17-lewm-evolution-results.md)
- [Every illustrative replay clip](REFERENCE.md#episode-gallery)
- [How to read the data files](../../DATA_GUIDE.md)
- [Previous experiment: recovering progress](../lewm16/README.md)
- [Back to the dataset introduction](../../README.md)

This environment's record contains {info['attempts']:,} valid attempts, including
engineering, fixed tuning, training, selection and final evaluation. `lewm17` is the
experiment number, not a new version of LeWM. No general safety guarantee or
compute-efficiency advantage follows from this comparison.
'''
    preserve_and_replace(src / 'README.md', dst, body)


def main_card(title):
    env = title.lower()
    before, after = (3, 34) if env == 'tworoom' else (0, 33)
    strict = '6-pixel' if env == 'tworoom' else '1.5 cm'
    task = ('A dot must move through a gap in a forbidden band to reach a target.'
            if env == 'tworoom' else
            'A robot arm must reach a target joint pose while keeping its fingertip outside a forbidden region.')
    body = f'''# {title}: helping a planner finish safely

**The question:** can we help an existing model reach its goal safely without retraining it?

{task}
LeWM predicts what possible moves would do. A planner chooses moves using those
predictions. A separate safety check tests the predicted path against the forbidden
region and an extra buffer. We call the checked planner SafeCEM.

## The clearest result

At a {strict} buffer on familiar layouts, better fixed planning settings raised safe
completion from **{before} of 36 attempts to {after} of 36**. The model and buffer were
unchanged. Every attempt moved, and neither checked group crossed either safety boundary.
The original settings usually got stuck; the tuned settings usually finished.

We then tested feedback and evolutionary search. **Neither established a clear added
advantage over the strong fixed settings.** Useful planning settings and a winning
optimizer are two different claims.

## Start with this video

[Watch the four methods on the same case](campaigns/lewm16/videos/comparisons/A-paired-high-margin.mp4)

![Original and tuned planning on one matched case](campaigns/lewm16/videos/comparisons/A-paired-high-margin.gif)

Read the four panels as a short story: `native` reaches the goal but breaches the buffer;
`canonical` has the safety check but does not finish; `fixed` finishes with better
constant settings; `adaptive` also finishes while allowed to change settings.
These are the outcomes of this example, not the full performance comparison.

{legend(title)}

## Follow the two experiments in order

| Read next | Question | Answer so far |
|---|---|---|
| [1. Recovering progress](campaigns/lewm16/README.md) | Can better planning settings help? | Yes on familiar layouts; fixed settings also perform strongly on new cases |
| [2. Testing evolution](campaigns/lewm17/README.md) | Can a learned 36-number selector do better? | The completed comparison does not establish an evolutionary advantage |

The folder names `lewm16` and `lewm17` are experiment numbers. They do not mean
16 or 17 versions of the learned model. The model weights remained fixed.

## Find the evidence you need

- **Understand the result:** read the two experiment pages above.
- **Browse more videos:** each experiment links its full replay gallery.
- **Read or load the data:** use the [data guide](DATA_GUIDE.md).
- **Inspect the earlier study:** use the [historical reference](REFERENCE.md#historical-campaign-description).
  Older work used different comparisons and monitoring; it is not pooled with the new tests.

The safety check uses true simulator state and dynamics. Zero observed breaches is
not a guarantee outside these tests. A wider buffer is a distance requirement, not
a calibrated probability of failure. Raw results, videos and frozen protocols remain
available beneath this guide.
'''
    preserve_and_replace(OUT/'snapshots'/title/'README.md', STAGE/title/'README.md',
                         body, historical=True)


def data_guide(title):
    return f'''# Reading the {title} evidence

[Back to the introduction](README.md)

## Choose the question first

`campaigns/lewm16/` is the progress-recovery study.
`campaigns/lewm17/` is the optimizer comparison.
The older root-level `data/` folder belongs to the earlier study, not these two.
The same distinction applies to the dataset viewer: choose a `lewm16_...` or
`lewm17_...` configuration for the newer experiments; `episodes` is historical.

## Then choose the level of detail

| File within a campaign | What one row or file means | Use it for |
|---|---|---|
| `data/episodes.parquet` or `episodes.jsonl` | One attempted episode and its result | Completion and failure counts |
| `data/trajectories.parquet` | One recorded action and resulting state | How an attempt unfolded |
| `data/dense-states.parquet` | A recorded physics state (Reacher), or action state (TwoRoom) | Checking boundary crossings between actions |
| `data/raw-attempts.tar.gz` | Exact original attempt folders | Full reproduction and auditing |
| `media-index.json` | One replay, its outcome and selection reason | Connecting a clip to its data |
| `protocol/` and `code/` | Frozen plans, reports and implementation | Checking how the experiment was run |
| `controllers/` in study 17 | Selected parameters and search records | Checking what each method actually deployed |

Parquet is a compact table format. JSONL is text with one JSON record per line.
A valid episode is a successfully recorded experiment, **not necessarily a successful task**.
A failed engineering attempt is kept in the archive but is not silently counted as a valid rollout.

## The result labels

| Label | Meaning |
|---|---|
| `requested_safe_completion` | Reached the goal within the allowance and never breached the requested buffer |
| `benchmark_safe_completion` | Reached the goal without entering the original forbidden region |
| `any_violation` | Entered the original forbidden region |
| `margin_breach` | Entered the extra forbidden buffer, even if the original danger was avoided |
| `unfinished` | Did not reach the goal within the attempt |
| `engaged` | Moved according to the experiment's movement check |
| `J` | Action count if safely successful; 50 otherwise, so failure is not rewarded as fast completion |

An attempt can be both unfinished and unsafe. Engagement does not mean useful progress.
For the checked final comparisons, all observed breaches were zero, so the remaining
failures were unfinished. A fraction such as 99/120 means 99 safe completions from
120 attempts; it does not mean 120 independent task geometries.

## Names you may see in files and video titles

| Name | Meaning |
|---|---|
| `canonical` | Original planning settings with the safety check |
| `fixed` | Tuned settings held constant during an attempt |
| `adaptive` | Feedback controller in the progress-recovery study |
| `historical` / `tuned` | Earlier fixed reference / newly retuned fixed reference in the optimizer study |
| `reinforce` | Policy-gradient learning |
| `r0` to `r4` | Five separate optimizer runs |
| `anchor` | A shared hand-designed fallback selected instead of updated coefficients |
| `seed` | Random-number setting; not a new task geometry |

Stage letters are local to each study: study 16 uses `A` for familiar-layout final
evaluation and `B` for new-case final evaluation. Study 17 uses `H` for final evaluation,
`T` for training, `V` for validation and `A` for anchor evaluation. Never join stages
across studies just because their letters match.

[Study 16 complete report](campaigns/lewm16/protocol/16-lewm-results.md) and
[study 17 complete report](campaigns/lewm17/protocol/17-lewm-evolution-results.md)
provide the exact definitions, denominators, uncertainty and resource accounting.
'''


def older_card(title):
    if title == 'PushT':
        body = '''# Push-T: can a safety check be trusted, and can the task still finish?

A pusher must move a T-shaped block onto a target. We add forbidden pillars along
the route. A learned world model predicts progress; a separate check tries to keep
the **pusher** out of the pillars. This is an earlier study, recorded as task 10.

**The lesson:** a safety setting only means what the checker can actually enforce.
The approximate checker sometimes misread the pusher's path. The exact-path checker
avoided all pillar entries in its 140 tested episodes, but often failed to finish.
None observed is not a general guarantee.

## Start with the checker comparison

[Watch one case checked three ways](videos/checks/G80-seed201-d10-three-checks.mp4).
The same requested 10-pixel buffer is evaluated using different checkers, alongside
the planner without a check. Look for the pusher crossing red and for whether the T
actually reaches the goal. A large buffer on paper does not fix an inaccurate checker.

[Then watch the exact checker at several buffer sizes](videos/dial-sweep/G80-seed201-exact-check.mp4).

![One case at several safety margins](videos/dial-sweep/G80-seed201-exact-check.gif)

Red marks the pillars, amber the added buffer, blue the pusher's path, and green the
T's target pose. A red ring marks a recorded pillar entry. These videos replay saved
states; they are illustrative examples. Some search clips were selected for good
outcomes, so a striking clip is not evidence of average performance.

## What did the search show?

The search tuned planning and safety-interface settings using an approximate checker.
Some settings did better on the layout used for search, but the average gain did not
carry to the other layout. Three optimizer runs per method did not settle whether
the evolutionary method, NSGA-II, was better than random search. The best setting on
an evaluation panel is also an optimistic summary, because it was chosen from several.
This study does not establish evolutionary superiority or a smooth risk dial.

## Where to go next

- [Detailed results, all videos and loading instructions](REFERENCE.md)
- [The original complete write-up](results/paper/safecem10-gate-dial-study.pdf)
- [Later TwoRoom study: recovering safe progress](https://huggingface.co/datasets/Sachioster/SafetyDial-TwoRoom)
- [Later Reacher study and controlled optimizer comparison](https://huggingface.co/datasets/Sachioster/SafetyDial-Reacher)

`data/episodes.parquet` has one row per episode; `data/trajectories.parquet` records
its path. `results/` contains analyses, `videos/` the replays, and `protocol/` the
experiment records. The older detailed wording is preserved as a historical reference;
its references to a safety "guarantee" should be read as zero observed breaches in
those tests, not proof of general safety.
'''
    else:
        body = '''# Cube environment: avoiding danger can still leave the arm stuck

A robot arm must move its gripper to a target position shown in a goal image.
Although the environment is called Cube, **this experiment evaluates a reach**,
not a complete cube-picking or stacking task. A forbidden band lies across the route.
This is an earlier study, recorded as task 14.

**The lesson:** the safety check mostly prevented entries into the band, but the
planner often stopped making useful progress. Changing the search method did not
establish an evolutionary advantage.

## Start with this video

[Watch one case at several safety margins](videos/dial-sweep/C1-seed201.mp4).

![The same reach at several safety margins](videos/dial-sweep/C1-seed201.gif)

The panels compare the planner without a check and the checked planner with increasing
buffers. Red is the forbidden band, amber is the extra buffer, blue is the gripper path
and green is the target. Watch the distance left to the goal at the end: avoiding red
is not enough if the arm never finishes. The clip replays saved states and was chosen
to illustrate the contrast, not to represent average performance.

## What happened across the tests?

Without the check, the arm reached the goal in 8 of 10 attempts on each of three layouts,
but entered the band in 23 of the 30 attempts. Across 200 checked attempts, one entered
the band. That reduction in crossings did not translate into reliable completion.
At buffers of 1 cm or more in the sweep, no checked attempt reached the goal.

This check approximates how commanded motion moves the gripper, so it can be wrong.
Its good crossing count is an observation, not a guarantee. Larger buffers made the
progress problem worse in this sweep.

The evolutionary search, NSGA-II, did not beat random search. Some random-search
settings worked well on the easier evaluation layout and poorly on the search layout.
That is evidence of sensitivity to layout, not a robust controller that generalizes.
There were three independent search runs per method; comparing all nine pairs does
not create nine independent replications. The layout withheld from search had already
appeared during development.

## Where to go next

- [Detailed results, complete replay gallery and loading instructions](REFERENCE.md)
- [Later TwoRoom study: can better planning recover progress?](https://huggingface.co/datasets/Sachioster/SafetyDial-TwoRoom)
- [Later Reacher study: the completed optimizer comparison](https://huggingface.co/datasets/Sachioster/SafetyDial-Reacher)

`data/episodes.parquet` contains one row per scored episode; `data/trajectories.parquet`
contains the recorded path. `results/` contains analyses, `videos/` the replays, and
`protocol/` the frozen experiment records. The complete historical dataset has
3,113 scored episodes, including setup, the margin sweep and search. That number is
not the size of one final comparison.
'''
    preserve_and_replace(OUT/'snapshots'/title/'README.md', STAGE/title/'README.md',
                         body, historical=True)


def finish_manifests(title):
    snap = OUT/'snapshots'/title
    stage = STAGE/title
    if title in ['TwoRoom','Reacher']:
        for task in [16,17]:
            rel = f'campaigns/lewm{task}'
            manifest = json.loads((snap/rel/'sha256-manifest.json').read_text())
            for p in (stage/rel).rglob('*'):
                if p.is_file() and p.name != 'sha256-manifest.json':
                    manifest[str(p.relative_to(stage/rel))] = {
                        'bytes':p.stat().st_size,'sha256':sha(p)}
            write(stage/rel/'sha256-manifest.json',json.dumps(manifest,indent=2))
    root_manifest = json.loads((snap/'sha256-manifest.json').read_text())
    for p in stage.rglob('*'):
        if p.is_file() and p != stage/'sha256-manifest.json':
            root_manifest[str(p.relative_to(stage))] = {'bytes':p.stat().st_size,'sha256':sha(p)}
    write(stage/'sha256-manifest.json',json.dumps(root_manifest,indent=2))


def check_links(title):
    stage = STAGE/title
    old = set(json.loads((OUT/'snapshots'/title/'remote.json').read_text())['files'])
    available = old | {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    checked = 0
    for p in stage.rglob('*.md'):
        # Preserved historical documents may contain pre-existing relative links.
        if 'history' in p.parts or p.name == 'REFERENCE.md':
            continue
        text = p.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','#')):
                continue
            target = unquote(target.split('#',1)[0])
            rel = posixpath.normpath(str(p.parent.relative_to(stage))+'/'+target)
            assert rel in available, (title,p.name,target,rel)
            checked += 1
        if p.name == 'README.md':
            old_path = OUT/'snapshots'/title/p.relative_to(stage)
            before = yaml.safe_load(split_card(old_path.read_text())[0].strip('-\n'))
            after = yaml.safe_load(split_card(text)[0].strip('-\n'))
            assert before == after, 'Dataset viewer metadata changed'
    return checked


def main():
    summaries=[]
    for title in ['TwoRoom','Reacher','PushT','Cube']:
        if title in ['TwoRoom','Reacher']:
            main_card(title)
            for task in [16,17]:
                campaign(title,task)
            write(STAGE/title/'DATA_GUIDE.md',data_guide(title))
        else:
            older_card(title)
        finish_manifests(title)
        links=check_links(title)
        files=[p for p in (STAGE/title).rglob('*') if p.is_file()]
        summaries.append({'dataset':title,'files':len(files),'new_links_checked':links,
                          'bytes':sum(p.stat().st_size for p in files),
                          'viewer_metadata':'unchanged','experimental_data':'unchanged'})
    write(OUT/'documentation-validation.json',json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()

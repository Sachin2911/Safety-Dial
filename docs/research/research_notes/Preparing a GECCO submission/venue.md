# Official GECCO submission requirements

## What is confirmed for 2027

### Takeaway
The official 2027 call and dates are now published. The repository's October note that dates remain unconfirmed is outdated.

### Cited Findings
- Mandatory full-paper abstract: 18 January 2027. Full paper and poster-only submission: 25 January. Notification: 19 March. Camera-ready: 9 April. Presenter registration: 10 May. [Official dates](https://gecco-2027.sigevo.org/Important%2BDates).
- Conference: Krakow, Poland, 12-16 July 2027, in person only. [Homepage](https://gecco-2027.sigevo.org/HomePage).
- Full papers: eight pages excluding references, abstract at most 200 words. Poster-only: four pages including references, with no separate advance abstract. Full papers enter main proceedings; posters enter companion proceedings. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).
- ACM sigconf format, anonymous review; supplements limited to 10 MB and paper PDF to 25 MB. [Submission instructions](https://gecco-2027.sigevo.org/Paper%2BSubmission%2BInstructions).

### Inferences
- Plan against 18 and 25 January, not the old provisional 19 and 26 January dates. An internal complete-draft deadline in early January leaves revision time; this is planning advice.

### Gaps
- The submission-instructions page retains stale 2026 camera-ready examples. Use the explicitly updated 2027 call for current policy and confirm final formatting instructions nearer submission.
- No deadline timezone appears on the official dates or call pages retrieved. Do not assume Anywhere on Earth.
- No verified 2027 student-workshop or late-breaking submission deadlines were obtained. They should not be promised as scheduled fallbacks. The poster route is confirmed.

## What reviewers require

### Takeaway
GECCO needs a distinct, relevant contribution supported by sound evidence. A new optimizer or universal superiority is not a stated submission requirement.

### Cited Findings
- Review considers significance, soundness, novelty, clarity, writing, relevance and reproducibility. Double-blind review is track-specific. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).
- New 2027 pilot: papers whose results centrally involve code are expected to provide runnable code reproducing main results; implementation details of the initiative are forthcoming. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).
- AI assistance in research execution must be detailed in Methods; writing assistance belongs in acknowledgments. Authors retain responsibility. Wholesale generated manuscript text is prohibited. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).
- Exact short excerpts from the research-execution policy: "Any use of AI tools to support the execution of the research" and "must be disclosed and described in detail in the Methods". The intervening parenthesis explicitly covers code implementation, experimental design/analysis, results generation and other research-lifecycle contributions relevant to conclusions. The following sentence includes code, datasets and AI-dependent charts/figures. The writing provision separately requires naming the tool and describing its use in acknowledgments; the prohibition covers wholesale text generation, especially but not exclusively literature reviews, and warns about hallucinated or misused references. These excerpts total 24 quoted words from the call; any downstream writer should avoid additional quotation from that page. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).

### Inferences
- For this project, reproducibility requires more than curated result tables: a reviewer-accessible code and asset route, environment pins, frozen configurations, scoring and analysis commands, and evidence sufficient to reconstruct the central results.
- The new AI policy materially affects this agent-assisted project. The author should maintain an accurate account of AI involvement and personally develop and verify the manuscript. This is an application of the official policy, not an assertion that the project is ineligible.
- A controlled empirical result can fit; merely assembling existing methods without a new finding would leave the novelty and significance questions unanswered.

### Gaps
- No official minimum number of environments, seeds or positive results is specified. Sample sizes must serve the scientific claim, not an invented venue threshold.
- The call simultaneously requires anonymous submission and writing-use disclosure in acknowledgments. Ensure anonymous Methods disclosure and recheck submission guidance for acknowledgment handling.

## Which track and fallback fit

### Takeaway
Neuroevolution is the most direct candidate if evolving compact neural adapters remains central. Evolutionary Machine Learning is a plausible alternative for the wider frozen-model adaptation contribution.

### Cited Findings
- Neuroevolution includes recurrent networks, evolutionary strategies, reinforcement learning, novel representations and efficiency, and evolutionary robotics. [2027 NE scope](https://gecco-2027.sigevo.org/Track?itemId=59).
- EML includes reinforcement learning, transfer and representation learning, application-focused papers, and strongly encourages appropriate comparisons with non-evolutionary ML methods. [2027 EML scope](https://gecco-2027.sigevo.org/Track?itemId=53).
- BBSR covers benchmarking and rigorous reproduction studies; reproduction papers must explain why findings differ and make artifacts public at submission. [2027 BBSR scope](https://gecco-2027.sigevo.org/Track?itemId=2672).
- Poster-only papers explicitly accommodate original work that is less mature or complete. [2027 call](https://gecco-2027.sigevo.org/Call%2Bfor%2BPapers).

### Inferences
- Select NE or EML based on the final central claim, in consultation with the supervisor. BBSR is suitable only if the contribution becomes a substantive benchmarking or reproducibility result, not simply because experiments have audits.
- Aim for the full paper while setting a December maturity checkpoint. A bounded, credible negative result can support an empirical paper if it explains a generalizable limitation; a collection of inconclusive pilots is a weaker full-paper case.

### Gaps
- Acceptance cannot be predicted from the current repository or guaranteed by satisfying a checklist.

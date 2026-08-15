Direction

Latent safety filters currently need supervised failure labels — a human marks what "unsafe" looks like, which is the cost-function specification problem in another form. This project asks whether the failure set can instead be derived from dynamics alone: a state is unsafe when its reachable future set collapses, i.e. when it is irrecoverable. That quantity is computable from a world model without labels. The work establishes reachability analysis in a JEPA latent (LeWM/SIGReg) — untested territory, since existing latent safety filters use reconstructive RSSM models — and then exposes the resulting conservatism as a runtime-adjustable parameter, the Safety Dial, letting an operator set caution at deployment rather than at training time.

Todo

Week 1 — prior art

 Read Nakamura et al., Latent Safety Filters (arXiv 2502.00935)
 Read Seo et al., UNISafe (arXiv 2505.00779)
 Read Agrawal et al., AnySafe (arXiv 2509.19555)
 Verify nobody has done reachability in a JEPA latent — OpenReview + arXiv listings, not just search
 Settle definition: empowerment vs recoverability. Pick one, write it down
 Skim LeJEPA/SIGReg primary source (arXiv 2511.08544)

Week 2 — go/no-go experiment

 Get a released LeWM checkpoint (don't train your own yet)
 Hand-pick recoverable vs irrecoverable states in Push-T
 Compute forward-rollout dispersion at each; report AUROC
 Check the confound: does dispersion track danger or just how far the agent can move?
 Check rollout depth confound: does the score climb with horizon regardless of state?
 Decision point: strong separation → ICLR path; weak → thesis path

Admin, in parallel

 Create OpenReview profile with Wits email (moderation delay otherwise)
 Confirm Geraud has a qualifying acceptance on his OpenReview profile
 One-page delta to Geraud on the direction change

If go

 Pick hazard environment (Safety-Gymnasium, not Push-T)
 Set up UNISafe head-to-head
 Draft abstract by 11 Sep — deadline is 18 Sep
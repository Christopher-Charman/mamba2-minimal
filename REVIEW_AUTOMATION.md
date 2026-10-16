# Automated distinct reviewer evidence gate

This verifies GitHub-native reviews. It does not generate a review or misstate the original author's approval as independent. It does not merge or deploy.

1. Activate a **separate** code-review GitHub account or app capable of submitting native APPROVED PR reviews on this repository.
2. Configure the repository variable INDEPENDENT_REVIEWERS with authorized external reviewer login(s), comma-separated. Without this, the gate fails closed.
3. Promote this workflow to trusted main through a separately authorized merge, then enforce both the evidence check and CPU test check using a no-bypass GitHub ruleset. A feature-branch workflow is not a trusted merge guard.
4. A new source commit invalidates earlier approvals automatically: only GitHub native APPROVED reviews with commit_id exactly equal to the latest PR head pass.
5. Review-only commenters (which may include AI review apps) do not qualify until a separately verified attestation adapter is approved.

The 3-minute standard Ubuntu CI job uses only stdlib Python and read-only GitHub tokens; no AI API spend, weights, caches, artifacts, or deployment. Independent reviewer installation requires the owner's GitHub authorization. GitHub's own rule for dismissing stale approvals should be configured on main at activation.

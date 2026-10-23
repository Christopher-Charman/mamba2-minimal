"""Fail-closed validator for a distinct GitHub-native PR approval."""
import json
import os
import re
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SHA = re.compile(r"^[0-9a-f]{40}$")
DENIED = {"github-actions[bot]", "github-actions", "dependabot[bot]"}


def who(person):
    login = person.get("login") if isinstance(person, dict) else None
    return login.casefold() if isinstance(login, str) else ""


def decide(pr, reviews, authorized, expected):
    def blocked(reason):
        return {"ok": False, "reason": reason, "head": expected}
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        return blocked("invalid_head")
    if not isinstance(pr, dict) or not isinstance(reviews, list):
        return blocked("malformed_evidence")
    if not isinstance(pr.get("head"), dict) or pr["head"].get("sha") != expected:
        return blocked("head_changed")
    if pr.get("state") != "open" or pr.get("draft") or pr.get("merged"):
        return blocked("not_open_reviewable")
    author = who(pr.get("user"))
    if not author:
        return blocked("missing_author")
    allowed = {x.strip().casefold() for x in authorized if isinstance(x, str) and x.strip()}
    allowed -= DENIED | {author}
    if not allowed:
        return blocked("independent_identity_not_configured")
    if len(reviews) >= 100:
        return blocked("review_page_incomplete")
    latest = {}
    for review in reviews:
        if not isinstance(review, dict):
            return blocked("malformed_review")
        name = who(review.get("user"))
        if name not in allowed:
            continue
        stamp, ident = review.get("submitted_at"), review.get("id")
        if not isinstance(stamp, str) or not stamp or not isinstance(ident, int):
            return blocked("invalid_review_record")
        if name not in latest or (stamp, ident) > latest[name][0]:
            latest[name] = ((stamp, ident), review)
    if not latest:
        return blocked("no_authorized_reviews")
    for name, (_, review) in latest.items():
        if review.get("state") == "CHANGES_REQUESTED":
            return blocked("changes_requested_by_" + name)
    for name, (_, review) in sorted(latest.items()):
        if review.get("state") == "APPROVED" and review.get("commit_id") == expected:
            return {"ok": True, "reason": "distinct_review_current_head",
                    "head": expected, "reviewer": name, "review_id": review["id"]}
    return blocked("no_current_head_approval")


def get(repo, endpoint, token):
    request = Request("https://api.github.com/repos/" + repo + "/" + endpoint,
                      headers={"Authorization": "Bearer " + token,
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28",
                               "User-Agent": "mamba2-independent-review-gate"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    number = os.environ.get("PR_NUMBER", "")
    head = os.environ.get("EXPECTED_HEAD_SHA", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    raw_allow = os.environ.get("AUTHORIZED_REVIEWERS", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        result = {"ok": False, "reason": "invalid_repository"}
    elif not number.isdecimal() or int(number) < 1 or not token:
        result = {"ok": False, "reason": "missing_api_context"}
    elif not raw_allow.strip():
        result = {"ok": False, "reason": "independent_identity_not_configured"}
    else:
        try:
            pr = get(repo, "pulls/" + str(int(number)), token)
            reviews = get(repo, "pulls/" + str(int(number)) + "/reviews?per_page=100", token)
            result = decide(pr, reviews, raw_allow.split(","), head)
        except (HTTPError, URLError, ValueError, TimeoutError) as exc:
            result = {"ok": False, "reason": "api_unavailable", "error": type(exc).__name__}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

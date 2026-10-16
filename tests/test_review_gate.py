import unittest
from review.review_gate import decide

A, B = "a"*40, "b"*40
def pr(head=A, author="Christopher-Charman", **overrides):
    d={"head":{"sha":head},"user":{"login":author},
       "state":"open","draft":False,"merged":False}
    d.update(overrides)
    return d
def rv(name="external-reviewer", state="APPROVED", commit=A, id=1):
    return {"user":{"login":name},"state":state,"commit_id":commit,
            "id":id,"submitted_at":"2026-10-16T19:00:00Z"}
class GateTests(unittest.TestCase):
    def check(self, reviews=None, author="Christopher-Charman", allow=None, head=A, **props):
        return decide(pr(head=head,author=author,**props),
                      [] if reviews is None else reviews,
                      ["external-reviewer"] if allow is None else allow, head)
    def test_positive_exact_commit(self):
        self.assertTrue(self.check([rv()])["ok"])
    def test_stale_commit_blocked(self):
        self.assertFalse(self.check([rv(commit=B)])["ok"])
    def test_no_approval(self):
        self.assertFalse(self.check()["ok"])
    def test_author_approval_cannot_pass(self):
        self.assertFalse(self.check([rv(name="Christopher-Charman")],allow=["Christopher-Charman"])["ok"])
    def test_actions_bot_cannot_pass(self):
        self.assertFalse(self.check([rv(name="github-actions[bot]")],allow=["github-actions[bot]"])["ok"])
    def test_unlisted_reviewer_cannot_pass(self):
        self.assertFalse(self.check([rv(name="someone-else")])["ok"])
    def test_comment_is_not_approval(self):
        self.assertFalse(self.check([rv(state="COMMENTED")])["ok"])
    def test_changes_requested_blocks(self):
        self.assertFalse(self.check([rv(),rv(state="CHANGES_REQUESTED",id=2)])["ok"])
    def test_newer_approval_clears_change_request(self):
        self.assertTrue(self.check([rv(state="CHANGES_REQUESTED"),rv(id=2)])["ok"])
    def test_dismissed_review_invalidates(self):
        self.assertFalse(self.check([rv(),rv(state="DISMISSED",id=2)])["ok"])
    def test_new_commit_invalidates(self):
        self.assertFalse(self.check([rv()],head=B)["ok"])
    def test_closed_rejected(self):
        self.assertFalse(self.check([rv()],state="closed")["ok"])
    def test_draft_rejected(self):
        self.assertFalse(self.check([rv()],draft=True)["ok"])
    def test_bad_sha_rejected(self):
        self.assertFalse(decide(pr(),[rv()],["external-reviewer"],"not-sha")["ok"])
    def test_incomplete_review_page_rejected(self):
        self.assertFalse(self.check([rv(id=i) for i in range(100)])["ok"])
    def test_wrong_author_rejected(self):
        self.assertFalse(self.check([rv()],author="")["ok"])
    def test_authorization_case_insensitive(self):
        self.assertTrue(self.check([rv(name="EXTERNAL-REVIEWER")])["ok"])
    def test_empty_reviewer_allowlist_rejected(self):
        self.assertFalse(self.check([rv()],allow=[])["ok"])
    def test_head_drift_before_gate(self):
        self.assertFalse(decide(pr(head=B),[rv()],["external-reviewer"],A)["ok"])
if __name__ == "__main__":
    unittest.main()

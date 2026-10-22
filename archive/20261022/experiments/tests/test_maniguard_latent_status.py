"""Started supplemental model calls stay charged even if no response is preserved."""

import json

from helpers.maniguard_latent_status import supplemental_decoder_accounting


def test_unreturned_and_duplicate_supplemental_requests_are_not_free(tmp_path):
    directory = tmp_path / "gripper-authority-20261020"
    directory.mkdir()
    rows = [{"id": "probe", "family": "jar"}, {"id": "probe", "family": "jar"}]
    (directory / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    result = supplemental_decoder_accounting(tmp_path)
    assert result["started_requests"] == 2 and result["remaining_requests"] == 46
    assert result["returned_receipts"] == 0 and result["by_family"]["jar"] == 2
    assert result["unresolved_request_ids"] == ["probe", "probe"]
    assert result["duplicate_request_ids"] == ["probe"]


def test_incomplete_tail_is_visible_and_returned_calls_remain_charged(tmp_path):
    directory = tmp_path / "gripper-authority-20261020"
    directory.mkdir()
    (directory / "requests.jsonl").write_text(
        json.dumps({"id": "probe", "family": "clutter"}) + '\n{"id":')
    (directory / "probe.json").write_text(
        json.dumps({"id": "probe", "server_receipt": {"status": "returned"}}))
    result = supplemental_decoder_accounting(tmp_path)
    assert result["started_requests"] == 1 and result["remaining_requests"] == 47
    assert result["returned_receipts"] == 1 and result["unresolved_request_ids"] == []
    assert result["incomplete_request_tail"] is True

"""Only the exact original deadline or explicitly authorized extension may arm."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/maniguard_latent_deadline.py"
spec = importlib.util.spec_from_file_location("latent_deadline", path)
deadline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deadline)


def test_original_and_authorized_extension_have_fixed_absolute_clocks():
    rental = {"instance_id": 56777435, "stop": {"deadline_utc": "2026-10-20T03:00:00Z"}}
    assert deadline.deadline_configuration(rental, "56777435") == (56777435, 1792465200)
    rental["stop"]["deadline_utc"] = "2026-10-20T05:00:00Z"
    with pytest.raises(ValueError, match="authorization"):
        deadline.deadline_configuration(rental, "56777435")
    rental["deadline_extension"] = {
        "previous_deadline_utc": "2026-10-20T03:00:00Z",
        "new_deadline_utc": "2026-10-20T05:00:00Z",
        "user_message": "Ok great, Also you can extend the time to 5 UTC"}
    assert deadline.deadline_configuration(rental, "56777435") == (56777435, 1792472400)
    with pytest.raises(ValueError, match="instance"):
        deadline.deadline_configuration(rental, "56777436")
    rental["stop"]["deadline_utc"] = "2026-10-20T06:00:00Z"
    with pytest.raises(ValueError, match="without authorization"):
        deadline.deadline_configuration(rental, "56777435")

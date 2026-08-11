"""Each structural-driver claim holds — one subtest per claim, named by its ID.

Mirrors the stochadex catalogue's claim<->test binding: the claims live in a
non-test module (``behaviour.py``) so they could also feed a card, and the test
just asserts each one's monotone direction.
"""

import pytest

from solarfleet.behaviour import observed_behaviour

CLAIMS = observed_behaviour()


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.id for c in CLAIMS])
def test_claim_holds(claim):
    assert claim.holds(), f"{claim.id}: {claim.observations}"


def test_all_six_claims_present():
    ids = {c.id for c in CLAIMS}
    assert ids == {
        "higher_cloud_volatility_raises_fleet_variability",
        "wider_geographic_dispersion_lowers_fleet_variability",
        "faster_cloud_reversion_lowers_output_variability",
        "steeper_tilt_shifts_output_toward_winter",
        "southward_orientation_raises_annual_output",
        "higher_latitude_widens_summer_winter_ratio",
    }

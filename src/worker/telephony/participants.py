"""SIP/telephony participant classifiers."""

from __future__ import annotations

from typing import Any


def _attributes(participant: Any) -> dict[str, str]:
    attrs = getattr(participant, "attributes", None) or {}
    if isinstance(attrs, dict):
        return attrs
    # Some SDK builds expose a mapping-like proxy
    try:
        return dict(attrs)
    except Exception:  # noqa: BLE001
        return {}


def _identity(participant: Any) -> str:
    return getattr(participant, "identity", "") or ""


def is_telephony_participant(participant: Any) -> bool:
    """
    A SIP participant is a phone leg. Outbound legs are `sip-out-...`; we also
    detect phone-mic track publications and the SIP callStatus attribute.
    """
    identity = _identity(participant)
    if identity.startswith(("sip", "telephony", "phone")):
        return True

    attrs = _attributes(participant)
    if "sip.callStatus" in attrs:
        return True

    track_pubs = getattr(participant, "track_publications", None) or getattr(
        participant, "trackPublications", None
    )
    if track_pubs is not None:
        try:
            values = track_pubs.values() if hasattr(track_pubs, "values") else track_pubs
        except Exception:  # noqa: BLE001
            values = []
        for pub in values:
            name = getattr(pub, "name", None) or (pub.get("name") if isinstance(pub, dict) else None)
            if name == "phone-mic":
                return True
    return False


def is_outbound_leg(participant: Any) -> bool:
    """Outbound legs join in `dialing` state; inbound legs are already connected."""
    identity = _identity(participant)
    if identity.startswith("sip-out"):
        return True
    attrs = _attributes(participant)
    return attrs.get("direction") == "outbound"

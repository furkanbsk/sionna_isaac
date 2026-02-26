from __future__ import annotations

from isaacsim_sionna.bridge.isaac_adapter import IsaacAdapter


def _cfg() -> dict:
    return {
        "runtime": {},
        "isaac": {
            "actor_motion": {
                "enabled": True,
                "backend": "procedural",
                "render_fallback": {
                    "enabled": True,
                    "use_if_missing_or_xform_only": True,
                    "prim_type": "Capsule",
                },
            },
            "prim_paths": {"actors": ["/World/humanoid_01"]},
        },
        "scenario": {},
        "project": {},
    }


def test_actor_fallback_applied_when_missing(monkeypatch) -> None:
    adapter = IsaacAdapter(_cfg())
    called = {"fallback": 0}
    monkeypatch.setattr(adapter, "_stage_has_prim", lambda _: False)
    monkeypatch.setattr(adapter, "_make_actor_visible_fallback", lambda p, i: called.__setitem__("fallback", called["fallback"] + 1))
    adapter._ensure_actor_prims()  # pylint: disable=protected-access
    assert called["fallback"] == 1


def test_actor_fallback_skipped_for_ira_backend(monkeypatch) -> None:
    cfg = _cfg()
    cfg["isaac"]["actor_motion"]["backend"] = "ira"
    cfg["isaac"]["actor_motion"]["ira"] = {"enabled": True}
    adapter = IsaacAdapter(cfg)
    called = {"fallback": 0}
    monkeypatch.setattr(
        adapter,
        "_make_actor_visible_fallback",
        lambda p, i: called.__setitem__("fallback", called["fallback"] + 1),
    )
    adapter._ensure_actor_prims()  # pylint: disable=protected-access
    assert called["fallback"] == 0


def test_actor_fallback_applied_when_non_renderable(monkeypatch) -> None:
    adapter = IsaacAdapter(_cfg())
    called = {"fallback": 0}
    monkeypatch.setattr(adapter, "_stage_has_prim", lambda _: True)
    monkeypatch.setattr(adapter, "_is_prim_renderable", lambda _: False)
    monkeypatch.setattr(adapter, "_make_actor_visible_fallback", lambda p, i: called.__setitem__("fallback", called["fallback"] + 1))
    adapter._ensure_actor_prims()  # pylint: disable=protected-access
    assert called["fallback"] == 1

"""
Attribution kwargs, matching the exact tag names the Cognocient proxy
already accepts as X-Cost-* headers (backend/app/proxy.py tag_header_map)
so a customer moving between the wrapper and the proxy uses the same
tagging model, not two different ones to learn.
"""

TAG_KWARGS = (
    "cognocient_feature",
    "cognocient_department",
    "cognocient_user",
    "cognocient_session",
    "cognocient_tier",
    "cognocient_project",
    "cognocient_gl_account",
    "cognocient_workload",
    "cognocient_outcome",
    "cognocient_environment",
    "cognocient_variant",
    "cognocient_run_id",
)


def pop_tags(kwargs: dict) -> dict:
    """Strips cognocient_* kwargs out of a call's kwargs and returns them
    mapped to CallReport field names, so the real SDK call underneath never
    sees (and never rejects) an argument it doesn't recognize."""
    tags = {}
    for key in TAG_KWARGS:
        if key in kwargs:
            value = kwargs.pop(key)
            name = key[len("cognocient_"):]
            tags["run_id" if name == "run_id" else f"tag_{name}"] = value
    return tags

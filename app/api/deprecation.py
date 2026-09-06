"""
Shared marker for the legacy Stack B endpoints.

`POST /api/v1/decision` is the single authoritative ChronoPace decision source.
The `/api/race`, `/api/energy`, `/api/overtake`, `/api/strategy` endpoints predate
the v1 pipeline and are served by the legacy Stack B engines. Their numbers do NOT
match the v1 DecisionSnapshot and must not be treated as authoritative.

Every response from those routes carries `Deprecation: true` + a `Link` to the
canonical endpoint, and they are flagged `deprecated` in the OpenAPI schema.
"""

from fastapi import Response

_CANONICAL = "/api/v1/decision"
_NOTE = (
    "Legacy Stack B endpoint - NOT authoritative. Use POST /api/v1/decision "
    "(or the WebSocket decision_snapshot) as the single source of truth."
)


def mark_deprecated(response: Response) -> None:
    """FastAPI dependency: stamps deprecation headers on the response."""
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f'<{_CANONICAL}>; rel="successor-version"'
    response.headers["Warning"] = f'299 - "{_NOTE}"'

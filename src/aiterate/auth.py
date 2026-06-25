from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jwt
from fastapi import Depends, HTTPException, Request, status

from aiterate.config import settings


class Role(StrEnum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


ROLE_LEVELS = {Role.VIEWER: 1, Role.EDITOR: 2, Role.ADMIN: 3}


@dataclass(frozen=True)
class CurrentUser:
    id: str
    role: Role


def get_current_user(request: Request) -> CurrentUser:
    if not settings.auth_enabled:
        return CurrentUser(id="local-user", role=Role.ADMIN)

    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
        )

    if settings.admin_api_key and token == settings.admin_api_key:
        return CurrentUser(id="admin-api-key", role=Role.ADMIN)

    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT auth is enabled but AIT_JWT_SECRET is not configured.",
        )

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        ) from exc

    role = Role(payload.get("role", Role.VIEWER.value))
    return CurrentUser(id=str(payload.get("sub") or "jwt-user"), role=role)


def require_role(minimum_role: Role):
    def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if ROLE_LEVELS[user.role] < ROLE_LEVELS[minimum_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{minimum_role.value} role required.",
            )
        return user

    return dependency

"""
Bizbozz JWT authentication middleware.

The iOS app and React widget both forward the bizbozz JWT they received
after signing in via Firebase.  We verify signature + claims here and
expose the decoded payload as a dependency.

JWT payload shape (from bizbozz docs):
  {
    "userId":            "<UUID>",          # bizbozz User.id
    "businessId":        "<UUID>",          # active Business.id
    "role":              "owner" | "staff",
    "gymOwnerProfileId": "<UUID>",
    "permissions":       ["body_scanner", ...],
    "aud":               "business-app",
    "iss":               "gymheros"
  }
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings

bearer_scheme = HTTPBearer()


class BizBozzClaims(BaseModel):
    userId: str
    businessId: str
    role: str
    gymOwnerProfileId: str | None = None
    permissions: list[str] = []


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> BizBozzClaims:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_business,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return BizBozzClaims(**payload)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing required fields",
        )


def require_permission(permission: str):
    """Factory that returns a dependency requiring a specific permission."""

    def _check(claims: BizBozzClaims = Depends(verify_jwt)) -> BizBozzClaims:
        if permission not in claims.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return claims

    return _check

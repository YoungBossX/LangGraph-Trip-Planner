"""POI相关API路由"""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from ...config import get_settings
from ...models.schemas import PhotoName
from ...services.unsplash_service import get_unsplash_service
from ..guards import PublicAPIError, get_client_ip, get_rate_limiter

router = APIRouter(prefix="/poi", tags=["POI"])
logger = logging.getLogger(__name__)

_RATE_LIMIT_MESSAGE = "Too many requests. Please retry later."
_PHOTO_LOOKUP_FAILED_MESSAGE = "Photo lookup failed. Please try again later."
_PHOTO_RATE_MARKER = "poi_photo_rate_checked"


async def _enforce_photo_rate_limit(request: Request) -> bool:
    state = request.scope.setdefault("state", {}) if hasattr(request, "scope") else None
    if state is not None and state.get(_PHOTO_RATE_MARKER):
        return True

    settings = get_settings()
    decision = await get_rate_limiter().consume(
        "poi-photo",
        get_client_ip(request),
        limit=settings.photo_rate_limit,
        window_seconds=settings.photo_rate_window_seconds,
    )
    if not decision.allowed:
        raise PublicAPIError(429, "RATE_LIMITED", _RATE_LIMIT_MESSAGE, decision.retry_after)
    if state is not None:
        state[_PHOTO_RATE_MARKER] = True
    return True


@router.get(
    "/photo",
    summary="获取景点图片",
    description="根据景点名称从Unsplash获取图片"
)
async def get_attraction_photo(
    request: Request,
    name: Annotated[PhotoName, Query(description="景点名称")],
    rate_checked: bool = Depends(_enforce_photo_rate_limit),
):
    """
    获取景点图片

    Args:
        name: 景点名称

    Returns:
        图片URL
    """
    if rate_checked is not True:
        await _enforce_photo_rate_limit(request)

    try:
        unsplash_service = get_unsplash_service()
        photo_url = await asyncio.to_thread(unsplash_service.get_photo_url, f"{name} China landmark")

        if not photo_url:
            photo_url = await asyncio.to_thread(unsplash_service.get_photo_url, name)

        return {
            "success": True,
            "message": "获取图片成功",
            "data": {
                "name": name,
                "photo_url": photo_url
            }
        }

    except Exception as exc:
        logger.exception("Photo lookup failed")
        raise PublicAPIError(502, "PHOTO_LOOKUP_FAILED", _PHOTO_LOOKUP_FAILED_MESSAGE) from exc

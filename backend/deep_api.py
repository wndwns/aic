"""深度模型 API 路由（独立模块，避免与 server.py 既有路由纠缠）。

在 `server.py` 里只加一行注册调用：

    from deep_api import register_deep_routes
    register_deep_routes(app)

路由一览：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/deep/status` | 模型状态、训练配置、可复现信息 |
| GET | `/api/deep/risk?month=YYYY-MM` | 26 县风险评分与分级（默认最新月） |
| GET | `/api/deep/attention?month=YYYY-MM` | 跨模态注意力在时间维的分布（热力图数据源） |
| GET | `/api/deep/attribution/{region_id}?month=YYYY-MM` | 通道遮挡归因 + 物理残差 |
| GET | `/api/deep/training` | 触发/重跑训练（幂等，已有 checkpoint 则直接返回） |
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def register_deep_routes(app: Any) -> None:
    """把深度模型相关路由挂到给定的 FastAPI 应用上。"""

    @app.get("/api/deep/status")
    def deep_status() -> dict:
        from deep_service import get_service
        try:
            return get_service().status()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/deep/training")
    def deep_training(force: bool = False) -> dict:
        from deep_service import get_service
        try:
            return get_service().ensure_trained(force=force)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/deep/risk")
    def deep_risk(month: str | None = None) -> dict:
        from deep_service import get_service
        try:
            return get_service().predict(month)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/deep/attention")
    def deep_attention(month: str | None = None) -> dict:
        from deep_service import get_service
        try:
            return get_service().attention(month)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/deep/fields")
    def deep_fields(month: str | None = None) -> dict:
        """26 县该月气象场原始值（温度/降水/降雪/雪深/辐射/风度风向/湿度）。"""
        from deep_service import get_service
        try:
            return get_service().fields(month)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/deep/attribution/{region_id}")
    def deep_attribution(region_id: str, month: str | None = None) -> dict:
        from deep_service import get_service
        try:
            return get_service().attribution(region_id, month)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

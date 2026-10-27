from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...agent.graph import run_guard
from ..deps import require_api_key
from ..schemas import InspectRequest, InspectResponse

router = APIRouter(prefix="/v1/guard", tags=["guard"], dependencies=[Depends(require_api_key)])


def to_response(st: dict) -> InspectResponse:
    d = st["decision"]
    return InspectResponse(
        request_id=st.get("request_id", ""), action=d["final_action"], status=st.get("status", "done"),
        output_text=st.get("output_text"), findings=st.get("findings", []), citations=d["citations"],
        rationale=d["rationale"], confidence=d["confidence"], laya=d.get("laya"),
        hitl_case_id=st.get("hitl_case_id"), latency_ms=round(st.get("latency_ms", 0.0), 2), trace=st.get("trace", []),
    )


@router.post("/inspect", response_model=InspectResponse)
async def inspect(req: InspectRequest) -> InspectResponse:
    """Run the full agent loop on one prompt, response or tool payload."""
    st = await run_in_threadpool(run_guard, req.model_dump())
    return to_response(st)

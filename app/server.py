import json
import logging
from typing import Optional

import grpc
import gateway_pb2
import gateway_pb2_grpc

from app.semantic_store import SemanticStore


logger = logging.getLogger(__name__)


class OptimizerServicer(gateway_pb2_grpc.OptimizerServicer):
    def __init__(self, semantic_store: Optional[SemanticStore] = None):
        self._semantic_store = semantic_store or SemanticStore()

    def OptimizePrompt(self, request, context):
        logger.info(
            "OptimizePrompt called for user_id=%s model_requested=%s",
            request.user_id,
            request.model_requested,
        )

        cached_response = self._semantic_store.find_similar(request.prompt)
        if cached_response is not None:
            try:
                logger.info("Semantic cache hit for user_id=%s", request.user_id)
                payload = json.loads(cached_response)
                return gateway_pb2.OptimizationResponse(
                    optimized_prompt=payload.get("optimized_prompt", request.prompt),
                    target_model=payload.get("target_model", request.model_requested),
                    should_use_oss=payload.get("should_use_oss", False),
                    cached_response=cached_response,
                )
            except json.JSONDecodeError:
                logger.warning("Invalid semantic cache entry for prompt, ignoring hit")

        return gateway_pb2.OptimizationResponse(
            optimized_prompt=request.prompt,
            target_model=request.model_requested,
            should_use_oss=False,
        )

    def UpdateCache(self, request, context):
        logger.info("UpdateCache called for prompt length=%s", len(request.prompt))

        try:
            json.loads(request.response_json)
        except json.JSONDecodeError:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("response_json must be valid JSON")
            return gateway_pb2.CacheUpdateResponse(success=False)

        self._semantic_store.add_to_cache(request.prompt, request.response_json)
        return gateway_pb2.CacheUpdateResponse(success=True)

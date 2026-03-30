import logging

import gateway_pb2
import gateway_pb2_grpc


logger = logging.getLogger(__name__)


class OptimizerServicer(gateway_pb2_grpc.OptimizerServicer):
    def OptimizePrompt(self, request, context):
        logger.info(
            "OptimizePrompt called for user_id=%s model_requested=%s",
            request.user_id,
            request.model_requested,
        )
        return gateway_pb2.OptimizationResponse(
            optimized_prompt=request.prompt,
            target_model=request.model_requested,
            should_use_oss=False,
        )

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from concurrent import futures

import grpc

import gateway_pb2_grpc
from app.server import OptimizerServicer


SERVER_ADDRESS = "[::]:50051"


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gateway_pb2_grpc.add_OptimizerServicer_to_server(OptimizerServicer(), server)
    server.add_insecure_port(SERVER_ADDRESS)
    server.start()
    logging.info("gRPC server listening on %s", SERVER_ADDRESS)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received, stopping gRPC server")
        server.stop(grace=5).wait()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    serve()

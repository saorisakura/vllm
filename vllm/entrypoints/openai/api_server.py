# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import gc
import importlib
import inspect
import json
import multiprocessing
import os
import signal
import socket
import tempfile
import uuid
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import Annotated, Any, Optional

import prometheus_client
import regex as re
import uvloop
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (load_chat_template,
                                         resolve_hf_chat_template,
                                         resolve_mistral_chat_template)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import (log_non_default_args,
                                              make_arg_parser,
                                              validate_parsed_serve_args)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              ClassificationRequest,
                                              ClassificationResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingChatRequest,
                                              EmbeddingCompletionRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoRAAdapterRequest,
                                              PoolingChatRequest,
                                              PoolingCompletionRequest,
                                              PoolingRequest, PoolingResponse,
                                              RerankRequest, RerankResponse,
                                              ScoreRequest, ScoreResponse,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              TranscriptionRequest,
                                              TranscriptionResponse,
                                              UnloadLoRAAdapterRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_classification import (
    ServingClassification)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.utils import (cli_env_setup, load_aware_call,
                                    with_cancellation)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (Device, FlexibleArgumentParser, get_open_zmq_ipc_path,
                        is_valid_ipv6_address, set_ulimit)
from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.version import __version__ as VLLM_VERSION

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        gc.collect()
        gc.freeze()
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    # engine_args is EngineArgs
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing,
            client_config) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    usage_context = UsageContext.OPENAI_API_SERVER
    # vllm.engine.arg_utils.py
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # V1 AsyncLLM.
    if envs.VLLM_USE_V1:
        if disable_frontend_multiprocessing:
            logger.warning(
                "V1 is enabled, but got --disable-frontend-multiprocessing. "
                "To disable frontend multiprocessing, set VLLM_USE_V1=0.")

        from vllm.v1.engine.async_llm import AsyncLLM
        async_llm: Optional[AsyncLLM] = None
        client_index = client_config.pop(
            "client_index") if client_config else 0
        try:
            async_llm = AsyncLLM.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats,
                client_addresses=client_config,
                client_index=client_index)

            # Don't keep the dummy data in memory
            await async_llm.reset_mm_cache()

            yield async_llm
        finally:
            if async_llm:
                async_llm.shutdown()

    # V0 AsyncLLM.
    elif (MQLLMEngineClient.is_unsupported_config(vllm_config)
          or disable_frontend_multiprocessing):

        engine_client: Optional[EngineClient] = None
        try:
            engine_client = AsyncLLMEngine.from_vllm_config(
                vllm_config=vllm_config,
                usage_context=usage_context,
                disable_log_requests=engine_args.disable_log_requests,
                disable_log_stats=engine_args.disable_log_stats)
            yield engine_client
        finally:
            if engine_client and hasattr(engine_client, "shutdown"):
                engine_client.shutdown()

    # V0MQLLMEngine.
    # mac cpu mode will run here
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.debug("Multiprocessing frontend to use %s for IPC Path.",
                     ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        # Ensure we can serialize transformer config before spawning
        maybe_register_config_serialize_by_value()

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        engine_process = context.Process(
            target=run_mp_engine,
            args=(vllm_config, UsageContext.OPENAI_API_SERVER, ipc_path,
                  engine_args.disable_log_stats,
                  engine_args.disable_log_requests, engine_alive))
        engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        def _cleanup_ipc_path():
            socket_path = ipc_path.replace("ipc://", "")
            if os.path.exists(socket_path):
                os.remove(socket_path)

        # Ensure we clean up the local IPC socket file on exit.
        atexit.register(_cleanup_ipc_path)

        # Build RPCClient, which conforms to EngineClient Protocol.
        build_client = partial(MQLLMEngineClient, ipc_path, vllm_config,
                               engine_pid)
        mq_engine_client = await asyncio.get_running_loop().run_in_executor(
            None, build_client)
        try:
            while True:
                try:
                    await mq_engine_client.setup()
                    break
                except TimeoutError:
                    if (not engine_process.is_alive()
                            or not engine_alive.value):
                        raise RuntimeError(
                            "Engine process failed to start. See stack "
                            "trace for the root cause.") from None

            yield mq_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mq_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(errors=[
            "Unsupported Media Type: Only 'application/json' is allowed"
        ])


router = APIRouter()


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def mount_metrics(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


def pooling(request: Request) -> Optional[OpenAIServingPooling]:
    return request.app.state.openai_serving_pooling


def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding


def score(request: Request) -> Optional[ServingScores]:
    return request.app.state.openai_serving_scores


def classify(request: Request) -> Optional[ServingClassification]:
    return request.app.state.openai_serving_classification


def rerank(request: Request) -> Optional[ServingScores]:
    return request.app.state.openai_serving_scores


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(
        content={'server_load': request.app.state.server_load_metrics})


@router.get("/ping", response_class=Response)
@router.post("/ping", response_class=Response)
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)


@router.post("/tokenize",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_IMPLEMENTED.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    try:
        generator = await handler.create_tokenize(request, raw_request)
    except NotImplementedError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_IMPLEMENTED.value,
                            detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    try:
        generator = await handler.create_detokenize(request, raw_request)
    except OverflowError as e:
        raise RequestValidationError(errors=[str(e)]) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.OK.value: {
                     "content": {
                         "text/event-stream": {}
                     }
                 },
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 }
             })
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.OK.value: {
                     "content": {
                         "text/event-stream": {}
                     }
                 },
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.NOT_FOUND.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API")

    try:
        generator = await handler.create_completion(request, raw_request)
    except OverflowError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value,
                            detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                            detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    handler = embedding(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Embeddings API")

    generator = await handler.create_embedding(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/pooling",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_pooling(request: PoolingRequest, raw_request: Request):
    handler = pooling(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Pooling API")

    generator = await handler.create_pooling(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, PoolingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/classify", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_classify(request: ClassificationRequest,
                          raw_request: Request):
    handler = classify(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Classification API")

    generator = await handler.create_classify(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ClassificationResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/score",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_score(request: ScoreRequest, raw_request: Request):
    handler = score(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Score API")

    generator = await handler.create_score(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, ScoreResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/score",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_score_v1(request: ScoreRequest, raw_request: Request):
    logger.warning(
        "To indicate that Score API is not part of standard OpenAI API, we "
        "have moved it to `/score`. Please update your client accordingly.")

    return await create_score(request, raw_request)


@router.post("/v1/audio/transcriptions",
             responses={
                 HTTPStatus.OK.value: {
                     "content": {
                         "text/event-stream": {}
                     }
                 },
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.UNPROCESSABLE_ENTITY.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def create_transcriptions(raw_request: Request,
                                request: Annotated[TranscriptionRequest,
                                                   Form()]):
    handler = transcription(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Transcriptions API")

    audio_data = await request.file.read()
    generator = await handler.create_transcription(audio_data, request,
                                                   raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, TranscriptionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/rerank",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
@load_aware_call
async def do_rerank(request: RerankRequest, raw_request: Request):
    handler = rerank(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Rerank (Score) API")
    generator = await handler.do_rerank(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, RerankResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/v1/rerank",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
async def do_rerank_v1(request: RerankRequest, raw_request: Request):
    logger.warning_once(
        "To indicate that the rerank API is not part of the standard OpenAI"
        " API, we have located it at `/rerank`. Please update your client "
        "accordingly. (Note: Conforms to JinaAI rerank API)")

    return await do_rerank(request, raw_request)


@router.post("/v2/rerank",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
@with_cancellation
async def do_rerank_v2(request: RerankRequest, raw_request: Request):
    return await do_rerank(request, raw_request)


TASK_HANDLERS: dict[str, dict[str, tuple]] = {
    "generate": {
        "messages": (ChatCompletionRequest, create_chat_completion),
        "default": (CompletionRequest, create_completion),
    },
    "embed": {
        "messages": (EmbeddingChatRequest, create_embedding),
        "default": (EmbeddingCompletionRequest, create_embedding),
    },
    "score": {
        "default": (RerankRequest, do_rerank)
    },
    "rerank": {
        "default": (RerankRequest, do_rerank)
    },
    "reward": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
    "classify": {
        "messages": (PoolingChatRequest, create_pooling),
        "default": (PoolingCompletionRequest, create_pooling),
    },
}

if envs.VLLM_SERVER_DEV_MODE:

    @router.get("/server_info")
    async def show_server_info(raw_request: Request):
        server_info = {"vllm_config": str(raw_request.app.state.vllm_config)}
        return JSONResponse(content=server_info)

    @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request: Request):
        """
        Reset the prefix cache. Note that we currently do not check if the
        prefix cache is successfully reset in the API server.
        """
        device = None
        device_str = raw_request.query_params.get("device")
        if device_str is not None:
            device = Device[device_str.upper()]
        logger.info("Resetting prefix cache with specific %s...", str(device))
        await engine_client(raw_request).reset_prefix_cache(device)
        return Response(status_code=200)

    @router.post("/sleep")
    async def sleep(raw_request: Request):
        # get POST params
        level = raw_request.query_params.get("level", "1")
        await engine_client(raw_request).sleep(int(level))
        # FIXME: in v0 with frontend multiprocessing, the sleep command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.post("/wake_up")
    async def wake_up(raw_request: Request):
        tags = raw_request.query_params.getlist("tags")
        if tags == []:
            # set to None to wake up all tags if no tags are provided
            tags = None
        logger.info("wake up the engine with tags: %s", tags)
        await engine_client(raw_request).wake_up(tags)
        # FIXME: in v0 with frontend multiprocessing, the wake-up command
        # is sent but does not finish yet when we return a response.
        return Response(status_code=200)

    @router.get("/is_sleeping")
    async def is_sleeping(raw_request: Request):
        logger.info("check whether the engine is sleeping")
        is_sleeping = await engine_client(raw_request).is_sleeping()
        return JSONResponse(content={"is_sleeping": is_sleeping})


@router.post("/invocations",
             dependencies=[Depends(validate_json_request)],
             responses={
                 HTTPStatus.BAD_REQUEST.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.UNSUPPORTED_MEDIA_TYPE.value: {
                     "model": ErrorResponse
                 },
                 HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                     "model": ErrorResponse
                 },
             })
async def invocations(raw_request: Request):
    """
    For SageMaker, routes requests to other handlers based on model `task`.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value,
                            detail=f"JSON decode error: {e}") from e

    task = raw_request.app.state.task

    if task not in TASK_HANDLERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: '{task}' for '/invocations'. "
            f"Expected one of {set(TASK_HANDLERS.keys())}")

    handler_config = TASK_HANDLERS[task]
    if "messages" in body:
        request_model, handler = handler_config["messages"]
    else:
        request_model, handler = handler_config["default"]

    # this is required since we lose the FastAPI automatic casting
    request = request_model.model_validate(body)
    return await handler(request, raw_request)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter",
                 dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest,
                                raw_request: Request):
        handler = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter",
                 dependencies=[Depends(validate_json_request)])
    async def unload_lora_adapter(request: UnloadLoRAAdapterRequest,
                                  raw_request: Request):
        handler = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


def load_log_config(log_config_file: Optional[str]) -> Optional[dict]:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load log config from file %s: error %s",
                       log_config_file, e)
        return None


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        err = ErrorResponse(message=exc.detail,
                            type=HTTPStatus(exc.status_code).phrase,
                            code=exc.status_code)
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request,
                                           exc: RequestValidationError):
        exc_str = str(exc)
        errors_str = str(exc.errors())

        if exc.errors() and errors_str and errors_str != exc_str:
            message = f"{exc_str} {errors_str}"
        else:
            message = exc_str

        err = ErrorResponse(message=message,
                            type=HTTPStatus.BAD_REQUEST.phrase,
                            code=HTTPStatus.BAD_REQUEST)
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if token := args.api_key or envs.VLLM_API_KEY:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if request.method == "OPTIONS":
                return await call_next(request)
            url_path = request.url.path
            if app.root_path and url_path.startswith(app.root_path):
                url_path = url_path[len(app.root_path):]
            if not url_path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    if args.enable_request_id_headers:
        logger.warning(
            "CAUTION: Enabling X-Request-Id headers in the API Server. "
            "This can harm performance at high QPS.")

        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = request.headers.get(
                "X-Request-Id") or uuid.uuid4().hex
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning("CAUTION: Enabling log response in the API Server. "
                       "This can include sensitive information and should be "
                       "avoided in production.")

        @app.middleware("http")
        async def log_response(request: Request, call_next):
            response = await call_next(request)
            response_body = [
                section async for section in response.body_iterator
            ]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            logger.info("response_body={%s}",
                        response_body[0].decode() if response_body else None)
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


async def init_app_state(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    model_config = vllm_config.model_config

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer=tokenizer,
                chat_template=None,
                tools=None,
                model_config=vllm_config.model_config,
            )

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\n"
                    "It is different from official chat template '%s'. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template, args.model)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await state.openai_serving_models.init_static_loras()
    # 下面每个方法都会有一个对应的server运行
    # OpenAIServing from vllm.entrypoints.openai.serving_engine.py
    # 下面每个class实现OpenAIServing的_preprocess、_build_response（in _pipeline）
    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None  # model_config.task default value
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    ) if model_config.runner_type == "generate" else None
    state.openai_serving_pooling = OpenAIServingPooling(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.runner_type == "pooling" else None
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    ) if model_config.task == "embed" else None
    state.openai_serving_scores = ServingScores(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger) if model_config.task in (
            "score", "embed", "pooling") else None
    state.openai_serving_classification = ServingClassification(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
    ) if model_config.task == "classify" else None
    state.jinaai_serving_reranking = ServingScores(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger
    ) if model_config.task == "score" else None
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
    )
    state.openai_serving_transcription = OpenAIServingTranscription(
        engine_client,
        model_config,
        state.openai_serving_models,
        request_logger=request_logger,
    ) if model_config.runner_type == "transcription" else None
    state.task = model_config.task

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def validate_api_server_args(args):
    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
            and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valid_tool_parses)} }})")

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.reasoning_parser \
        and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})")


def setup_server(args):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""

    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    addr, port = sock_addr
    is_ssl = args.ssl_keyfile and args.ssl_certfile
    host_part = f"[{addr}]" if is_valid_ipv6_address(
        addr) else addr or "0.0.0.0"
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

    return listen_address, sock


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(listen_address,
                            sock,
                            args,
                            client_config=None,
                            **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs['log_config'] = log_config

    async with build_async_engine_client(args, client_config) as engine_client:  # AsyncLLM/MQLLMEngineClient
        app = build_app(args)

        vllm_config = await engine_client.get_vllm_config()  # AsyncLLM.from_vllm_config
        # app.state is a dict for FastAPI app
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index,
                    listen_address)
        # vllm.entrypoints.launcher.py:serve_http
        """
关于await serve_http调用卡住的分析
在你提供的代码中，await serve_http本身不会 "卡住"，但它会阻塞当前协程直到 HTTP 服务器停止运行。这是因为serve_http函数内部启动了一个 Uvicorn 服务器，该服务器会持续监听网络请求直到被显式关闭。
代码执行流程分析
serve_http函数的核心逻辑：
python
运行
server = uvicorn.Server(config)
server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

try:
    await server_task  # 等待服务器任务完成（直到服务器被关闭）
except asyncio.CancelledError:
    return server.shutdown()  # 处理取消信号，优雅关闭服务器

服务器的关闭机制：
通过信号处理（SIGINT/SIGTERM）触发signal_handler
signal_handler中取消server_task，从而触发server.shutdown()
watchdog 任务的作用：
python
运行
watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))

该任务可能用于监控服务器状态或处理超时，但代码中未展示其具体实现。
可能导致 "卡住" 的原因
正常行为：
如果服务器启动成功并正常运行，await serve_http会持续阻塞，这是预期行为。
服务器会一直运行，直到接收到终止信号（如 Ctrl+C）。
异常情况：
端口被占用：如果指定的端口已被其他进程使用，Uvicorn 可能无法启动，但代码中的错误处理可能不够完善。
死锁或无限循环：在watchdog_loop或其他后台任务中可能存在逻辑错误，导致程序无法正常响应终止信号。
资源不足：系统资源（如内存、文件描述符）耗尽可能导致服务器无法正常运行。
信号处理问题：
如果信号处理函数（signal_handler）未能正确触发，服务器将无法接收到关闭指令。
调试建议
检查端口占用：
bash
lsof -i :<端口号>  # 替换为你配置的端口号

添加调试日志：
在serve_http函数中增加日志，确认服务器是否成功启动：
python
运行
logger.info("Starting HTTP server on port %s", uvicorn_kwargs["port"])
server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

try:
    logger.debug("Awaiting server task...")
    await server_task
    logger.debug("Server task completed.")
except asyncio.CancelledError:
    logger.info("Server task cancelled, initiating shutdown...")
    return server.shutdown()



检查 watchdog 任务：
确认watchdog_loop函数的实现，确保它不会导致死锁或阻塞主线程。
测试信号处理：
在服务器运行时尝试发送终止信号（Ctrl+C 或kill命令），观察日志输出。
总结
await serve_http并非 "卡住"，而是正常等待 HTTP 服务器运行。如果服务器无法启动或无法正常关闭，可能是由于端口冲突、信号处理问题或其他异常情况导致的。通过添加详细的调试日志和检查系统资源，可以进一步定位问题所在。

Uvicorn + FastAPI 启动示例
以下是一个使用 uvicorn.Config 和 uvicorn.Server 手动启动 FastAPI 应用的完整示例，包含异步启动、信号处理和优雅关闭：
python
运行
import asyncio
import signal
import uvicorn
from fastapi import FastAPI
from uvicorn.config import LOGGING_CONFIG

# 创建FastAPI应用
app = FastAPI(title="My FastAPI App", version="1.0.0")

# 定义API路由
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, query: str = None):
    return {"item_id": item_id, "query": query}

# 自定义日志配置（可选）
custom_log_config = LOGGING_CONFIG.copy()
custom_log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s %(message)s"

async def run_server():
    # 使用uvicorn.Config和uvicorn.Server异步启动FastAPI应用
    # 创建Uvicorn配置
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_config=custom_log_config,  # 使用自定义日志配置
        log_level="info",
        reload=True,  # 开发模式下启用自动重载
        workers=1,    # 单工作线程
        timeout_keep_alive=5,  # 保持连接超时时间（秒）
    )
    
    # 创建Uvicorn服务器实例
    server = uvicorn.Server(config)
    
    # 获取当前事件循环
    loop = asyncio.get_running_loop()
    
    # 设置信号处理
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(server, signal_name=s.name))
        )
    
    # 启动服务器
    print("Starting server on http://0.0.0.0:8000")
    await server.serve()

async def shutdown(server: uvicorn.Server, signal_name: str):
    # 处理服务器关闭信号
    print(f"Received shutdown signal: {signal_name}")
    server.should_exit = True
    await server.shutdown()
    print("Server shutdown complete")

# 主入口点
if __name__ == "__main__":
    asyncio.run(run_server())
进阶示例：多服务并行启动
以下示例展示如何同时启动多个服务（如 HTTP 和 WebSocket）：
python
运行
import asyncio
import signal
import uvicorn
from fastapi import FastAPI
from fastapi.websockets import WebSocket

# 创建FastAPI应用
app = FastAPI()

# HTTP路由
@app.get("/")
async def root():
    return {"message": "Hello from HTTP"}

# WebSocket路由
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")

# 自定义日志配置
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
    },
}

async def run_http_server():
    # 运行HTTP服务器
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_config=log_config,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    return server

async def run_ws_server():
    # 运行WebSocket服务器
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8001,
        log_config=log_config,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    return server

async def main():
    # 主函数：并行启动多个服务器
    http_server = await run_http_server()
    ws_server = await run_ws_server()
    
    # 创建服务器任务
    http_task = asyncio.create_task(http_server.serve())
    ws_task = asyncio.create_task(ws_server.serve())
    
    # 设置信号处理
    loop = asyncio.get_running_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    
    for s in signals:
        loop.add_signal_handler(
            s, 
            lambda s=s: asyncio.create_task(handle_shutdown(s.name, http_server, ws_server))
        )
    
    # 等待所有服务器任务完成
    await asyncio.gather(http_task, ws_task)

async def handle_shutdown(signal_name: str, *servers):
    # 处理多个服务器的关闭
    print(f"Received {signal_name}, initiating shutdown...")
    for server in servers:
        server.should_exit = True
        await server.shutdown()
    print("All servers shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
关键参数说明
参数	作用
app	FastAPI 应用实例
host	监听地址
port	监听端口
log_config	自定义日志配置
log_level	日志级别（debug, info, warning, error, critical）
reload	是否启用自动重载（开发模式）
workers	工作线程数
timeout_keep_alive	保持连接超时时间（秒）
lifespan	生命周期管理（on, off）
ssl_keyfile	SSL 私钥文件路径（启用 HTTPS）
ssl_certfile	SSL 证书文件路径（启用 HTTPS）
运行方式
直接运行 Python 文件：
bash
python main.py

生产环境建议：
使用进程管理器（如 systemd、supervisor）或容器化部署，并设置适当的重启策略。
验证服务：
bash
curl http://localhost:8000


这些示例展示了如何使用 uvicorn.Config 和 uvicorn.Server 手动启动 FastAPI 应用，提供了更灵活的配置选项和更精细的控制能力。
        """
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))

"""
使用 uvicorn.Server 启动多 worker 的 FastAPI 应用
要使用uvicorn.Config和uvicorn.Server启动多 worker 的 FastAPI 应用，需要注意以下几点：
多进程模式限制：直接通过uvicorn.Server启动多 worker 时，需要使用uvicorn.Worker类或结合asyncio的进程管理
主进程管理：每个 worker 都是独立进程，需要主进程协调管理
信号处理：需要正确处理跨进程的信号传播
以下是几种实现方式：
方法一：使用 uvicorn.Worker（推荐方式）
这是官方推荐的方式，通过 Gunicorn-like 的 Worker 类来管理多进程：
python
运行
import asyncio
import signal
from uvicorn import Config, Server
from uvicorn.workers import UvicornWorker

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}

# 自定义Worker类，支持异步启动
class CustomUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {"loop": "asyncio", "http": "auto"}

async def run_multi_workers():
    # 创建配置
    config = Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # 设置worker数量
        loop="asyncio",
        log_level="info",
    )
    
    # 创建服务器
    server = Server(config)
    
    # 设置信号处理
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        asyncio.get_running_loop().add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(server, signal_name=s.name))
        )
    
    # 启动服务器
    await server.serve()

async def shutdown(server: Server, signal_name: str):
    print(f"Received shutdown signal: {signal_name}")
    server.should_exit = True
    await server.shutdown()

if __name__ == "__main__":
    asyncio.run(run_multi_workers())
运行命令：
bash
python main.py
方法二：手动管理多进程（高级方式）
通过asyncio.create_subprocess_exec手动启动多个 worker 进程：
python
运行
import asyncio
import signal
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}

async def run_worker(worker_id: int):
    # 运行单个worker进程
    cmd = [
        "python",
        "-m",
        "uvicorn",
        "main:app",  # 替换为你的应用导入路径
        "--host", "0.0.0.0",
        "--port", str(8000 + worker_id),  # 每个worker使用不同端口
        "--workers", "1",  # 每个进程1个worker
        "--log-level", "info",
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    print(f"Worker {worker_id} started with PID {process.pid}")
    return process

async def run_multi_workers(num_workers: int = 4):
    # 启动多个worker进程
    workers = []
    
    # 启动所有worker
    for i in range(num_workers):
        worker = await run_worker(i)
        workers.append(worker)
    
    # 设置信号处理
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        asyncio.get_running_loop().add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown_workers(workers, signal_name=s.name))
        )
    
    # 等待所有worker完成
    await asyncio.gather(*[w.wait() for w in workers])

async def shutdown_workers(workers, signal_name: str):
    # 关闭所有worker进程
    print(f"Received {signal_name}, shutting down {len(workers)} workers...")
    
    # 发送终止信号
    for worker in workers:
        if worker.returncode is None:
            worker.terminate()
    
    # 等待所有worker退出
    await asyncio.gather(*[w.wait() for w in workers])
    print("All workers shutdown complete")

if __name__ == "__main__":
    asyncio.run(run_multi_workers())
运行命令：
bash
python main.py
方法三：结合 Gunicorn 和 UvicornWorker（生产环境推荐）
这是生产环境最常用的方式，通过 Gunicorn 管理 Uvicorn worker：
python
运行
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}
运行命令：
bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app -b 0.0.0.0:8000
参数说明：
-k uvicorn.workers.UvicornWorker：使用 Uvicorn 的 worker 类
-w 4：启动 4 个 worker 进程
main:app：应用导入路径
-b 0.0.0.0:8000：绑定地址和端口
关键配置参数
参数	说明
workers	启动的 worker 进程数量，通常设置为 CPU 核心数的 1-2 倍
loop	事件循环类型（asyncio/uvloop）
http	HTTP 协议实现（auto/h11/httptools）
timeout_keep_alive	保持连接超时时间（秒）
limit_concurrency	每个 worker 的最大并发请求数
reload	开发模式下启用自动重载（不建议在生产环境使用）
多 worker 注意事项
状态管理：每个 worker 是独立进程，内存不共享
避免在进程内存储全局状态
使用外部存储（Redis、数据库）共享状态
端口绑定：
方法一和方法三会自动处理端口共享
方法二需要为每个 worker 分配不同端口，再通过负载均衡器统一对外服务
日志管理：
配置集中式日志收集
避免多个 worker 写入同一个日志文件
健康检查：
添加健康检查端点（如/health）
使用负载均衡器监控 worker 状态
选择适合你场景的方法，方法一和方法三适合生产环境，方法二提供了更灵活的自定义能力。

FastAPI 多进程应用的信号处理详解
在手动启动多进程 FastAPI 应用时，正确处理跨进程的信号传播至关重要。以下是一个完整示例，展示如何在使用uvicorn.Server和asyncio管理多进程时，实现优雅的信号处理：
完整示例代码
python
运行
import asyncio
import signal
import os
import uvicorn
from fastapi import FastAPI

# 创建FastAPI应用
app = FastAPI(title="Multi-Worker FastAPI App")

# 定义API路由
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}

# 全局变量跟踪所有子进程
child_processes = []

async def run_worker(worker_id: int, port: int):
    # 运行单个worker进程
    cmd = [
        "python",
        "-m",
        "uvicorn",
        "main:app",  # 替换为你的应用导入路径
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
        "--log-level", "info",
        "--no-access-log",  # 减少日志量
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    # 记录子进程
    child_processes.append(process)
    print(f"Worker {worker_id} (PID: {process.pid}) started on port {port}")
    
    # 捕获子进程输出（可选）
    async def log_output(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"[{prefix}] {line.decode().strip()}")
    
    # 启动日志记录任务
    asyncio.create_task(log_output(process.stdout, f"Worker-{worker_id}-stdout"))
    asyncio.create_task(log_output(process.stderr, f"Worker-{worker_id}-stderr"))
    
    return process

async def monitor_workers(workers):
    # 监控所有worker进程，发现退出时重启
    while True:
        await asyncio.sleep(1)
        for i, worker in enumerate(workers):
            if worker.returncode is not None:
                print(f"Worker {i} (PID: {worker.pid}) exited with code {worker.returncode}")
                # 重启退出的worker
                workers[i] = await run_worker(i, 8000 + i)

async def handle_signal(signal_name: str):
    # 处理接收到的信号
    print(f"主进程收到信号 {signal_name}，开始优雅关闭...")
    
    # 向所有子进程发送终止信号
    for process in child_processes:
        if process.returncode is None:
            process.terminate()
            print(f"已向Worker (PID: {process.pid}) 发送终止信号")
    
    # 等待所有子进程退出
    await asyncio.gather(*[p.wait() for p in child_processes if p.returncode is None])
    print("所有Worker已优雅关闭")
    
    # 退出主进程
    os._exit(0)

async def main():
    # 主函数：启动多个worker并管理信号
    num_workers = 4  # worker数量
    
    # 启动所有worker
    workers = []
    for i in range(num_workers):
        worker = await run_worker(i, 8000 + i)
        workers.append(worker)
    
    # 设置信号处理
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)
    loop = asyncio.get_running_loop()
    
    for s in signals:
        loop.add_signal_handler(
            s, 
            lambda s=s: asyncio.create_task(handle_signal(s.name))
        )
    
    print(f"主进程 (PID: {os.getpid()}) 已启动，管理 {num_workers} 个Worker")
    
    # 启动worker监控任务
    monitor_task = asyncio.create_task(monitor_workers(workers))
    
    # 等待监控任务完成（理论上不会完成，除非被信号中断）
    await monitor_task

if __name__ == "__main__":
    asyncio.run(main())
信号处理关键点解析
信号注册与处理：
python
运行
signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)
for s in signals:
    loop.add_signal_handler(
        s, 
        lambda s=s: asyncio.create_task(handle_signal(s.name))
    )

注册常见终止信号（SIGHUP、SIGTERM、SIGINT、SIGQUIT）
使用asyncio.create_task异步处理信号，避免阻塞主事件循环
跨进程信号传播：
python
运行
async def handle_signal(signal_name: str):
    # 向所有子进程发送终止信号
    for process in child_processes:
        if process.returncode is None:
            process.terminate()  # 发送SIGTERM
            
    # 等待所有子进程退出
    await asyncio.gather(*[p.wait() for p in child_processes if p.returncode is None])

使用process.terminate()向子进程发送 SIGTERM 信号
等待所有子进程通过p.wait()完成优雅关闭
子进程监控与重启：
python
运行
async def monitor_workers(workers):
    while True:
        await asyncio.sleep(1)
        for i, worker in enumerate(workers):
            if worker.returncode is not None:
                # 重启退出的worker
                workers[i] = await run_worker(i, 8000 + i)

定期检查子进程状态
自动重启异常退出的 worker，保持系统可用性
测试信号处理
启动应用：
bash
python main.py

发送终止信号：
bash
# 使用Ctrl+C (SIGINT)
# 或发送SIGTERM
kill -TERM <主进程PID>

预期输出：
plaintext
主进程收到信号 SIGTERM，开始优雅关闭...
已向Worker (PID: 1234) 发送终止信号
已向Worker (PID: 1235) 发送终止信号
已向Worker (PID: 1236) 发送终止信号
已向Worker (PID: 1237) 发送终止信号
[Worker-0-stdout] INFO:     Shutting down
[Worker-1-stdout] INFO:     Shutting down
[Worker-2-stdout] INFO:     Shutting down
[Worker-3-stdout] INFO:     Shutting down
所有Worker已优雅关闭

生产环境建议
使用进程管理器：
对于生产环境，推荐使用 systemd 或 supervisor 等进程管理器
它们提供更完善的进程监控、重启和信号处理机制
添加健康检查：
python
运行
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


配合负载均衡器定期检查应用健康状态
日志集中化：
使用集中式日志系统（如 ELK Stack、Graylog）收集所有 worker 的日志
避免多个进程写入同一个日志文件导致冲突
平滑升级：
实现零停机部署策略，支持滚动升级
利用 Uvicorn 的--reload机制在开发阶段快速迭代
通过这种方式，你可以确保在多进程 FastAPI 应用中正确处理信号，实现优雅关闭和高可用性。
"""
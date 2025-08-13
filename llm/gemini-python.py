# -*- coding: utf-8 -*-
"""
Gemini API Python Proxy with Robust Streaming Retry and Standardized Error Responses.

To run this script:
1. Install necessary libraries:
   pip install "fastapi[all]" httpx
2. Run the server:
   uvicorn your_script_name:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Set

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

# --- Configuration ---
# Mirroring the JS CONFIG object
CONFIG = {
    "upstream_url_base": "https://api-proxy.me/gemini",
    "max_consecutive_retries": 3,
    "debug_mode": True,
    "retry_delay_ms": 750,
    "swallow_thoughts_after_retry": True,
}

# --- Constants ---
NON_RETRYABLE_STATUSES: Set[int] = {400, 401, 403, 404, 429}
FINAL_PUNCTUATION: Set[str] = {'.', '?', '!', '。', '？', '！', '}', ']', ')', '"', "'", '”', '’', '`', '\n'}
SSE_ENCODER = str.encode

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG if CONFIG["debug_mode"] else logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Global HTTPX Client ---
# Declared at the module level to be accessible by all functions.
# It will be initialized in the `startup_event`.
async_client: Optional[httpx.AsyncClient] = None

# --- Application Lifecycle Events (on_startup / on_shutdown) ---
# This is an older but more compatible way to manage resources like the HTTP client.
async def startup_event():
    """Initializes the HTTPX client when the application starts."""
    global async_client
    async_client = httpx.AsyncClient(timeout=300)
    logging.info("HTTPX AsyncClient started.")

async def shutdown_event():
    """Closes the HTTPX client when the application shuts down."""
    if async_client:
        await async_client.close()
        logging.info("HTTPX AsyncClient closed.")

app = FastAPI(on_startup=[startup_event], on_shutdown=[shutdown_event])

# --- CORS Middleware ---
# Replicates the handleOPTIONS and Access-Control-Allow-Origin logic
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Goog-Api-Key"],
)


# --- Helper Functions ---

def status_to_google_status(code: int) -> str:
    """Converts HTTP status code to Google's RPC status string."""
    status_map = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        429: "RESOURCE_EXHAUSTED",
        500: "INTERNAL",
        503: "UNAVAILABLE",
        504: "DEADLINE_EXCEEDED",
    }
    return status_map.get(code, "UNKNOWN")

async def standardize_initial_error(initial_response: httpx.Response) -> JSONResponse:
    """Creates a standardized JSON error response from an upstream error."""
    upstream_text = ""
    try:
        # Ensure the body is read before creating the error response
        await initial_response.aread()
        upstream_text = initial_response.text
        logging.error(f"Upstream error body (truncated): {upstream_text[:2000]}")
    except Exception as e:
        logging.error(f"Failed to read upstream error text: {e}")

    standardized = None
    if upstream_text:
        try:
            parsed = json.loads(upstream_text)
            if isinstance(parsed.get("error"), dict) and isinstance(parsed["error"].get("code"), int):
                if "status" not in parsed["error"]:
                    parsed["error"]["status"] = status_to_google_status(parsed["error"]["code"])
                standardized = parsed
        except (json.JSONDecodeError, AttributeError):
            pass

    if not standardized:
        code = initial_response.status_code
        message = "Resource has been exhausted (e.g. check quota)." if code == 429 else (initial_response.reason_phrase or "Request failed")
        status = status_to_google_status(code)
        details = [{"@type": "proxy.upstream", "upstream_error": upstream_text[:8000]}] if upstream_text else None
        standardized = {
            "error": {
                "code": code,
                "message": message,
                "status": status,
                "details": details,
            }
        }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Goog-Api-Key",
    }
    if "Retry-After" in initial_response.headers:
        headers["Retry-After"] = initial_response.headers["Retry-After"]

    return JSONResponse(
        content=standardized,
        status_code=initial_response.status_code,
        headers=headers
    )

async def write_sse_error_from_upstream(upstream_resp: httpx.Response) -> str:
    """Formats an upstream error into an SSE 'error' event string."""
    std_resp = await standardize_initial_error(upstream_resp)
    text_content = std_resp.body.decode("utf-8")
    
    retry_after = upstream_resp.headers.get("Retry-After")
    if retry_after:
        try:
            obj = json.loads(text_content)
            obj["error"]["details"] = (obj["error"].get("details") or []) + [{"@type": "proxy.retry", "retry_after": retry_after}]
            text_content = json.dumps(obj)
        except json.JSONDecodeError:
            pass
            
    return f"event: error\ndata: {text_content}\n\n"

async def sse_line_iterator(response: httpx.Response) -> AsyncGenerator[str, None]:
    """Yields lines from an SSE stream."""
    buffer = ""
    line_count = 0
    logging.debug("Starting SSE line iteration")
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8")
        lines = buffer.splitlines()
        buffer = lines.pop() if lines and not buffer.endswith(('\n', '\r')) else ""
        for line in lines:
            if line.strip():
                line_count += 1
                logging.debug(f"SSE Line {line_count}: {line[:200]}")
                yield line
    if buffer.strip():
        logging.debug(f"SSE stream ended. Yielding final buffer: \"{buffer.strip()}\"")
        yield buffer.strip()
    logging.debug(f"SSE stream ended. Total lines processed: {line_count}.")


def is_data_line(line: str) -> bool:
    return line.startswith("data: ")

def is_blocked_line(line: str) -> bool:
    return "blockReason" in line

def extract_finish_reason(line: str) -> str | None:
    """Extracts finishReason from a data line."""
    if "finishReason" not in line:
        return None
    try:
        i = line.find("{")
        if i == -1:
            return None
        data = json.loads(line[i:])
        fr = data.get("candidates", [{}])[0].get("finishReason")
        if fr:
            logging.debug(f"Extracted finishReason: {fr}")
        return fr
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logging.debug(f"Failed to extract finishReason from line: {e}")
        return None

def parse_line_content(line: str) -> Dict[str, Any]:
    """Parses text and thought status from a data line."""
    try:
        json_str = line[line.find('{'):]
        data = json.loads(json_str)
        part = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
        if not part:
            return {"text": "", "is_thought": False}

        text = part.get("text", "")
        is_thought = part.get("thought") is True

        if is_thought:
            logging.debug("Extracted thought chunk. This will be tracked.")
        elif text:
            logging.debug(f"Extracted text chunk ({len(text)} chars): {text[:100]}")

        return {"text": text, "is_thought": is_thought}
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logging.debug(f"Failed to parse content from data line: {e}")
        return {"text": "", "is_thought": False}

def build_retry_request_body(original_body: Dict, accumulated_text: str) -> Dict:
    """Constructs a new request body for a retry attempt."""
    logging.debug(f"Building retry request body. Accumulated text length: {len(accumulated_text)}")
    logging.debug(f"Accumulated text preview: {accumulated_text[:200]}")
    
    retry_body = json.loads(json.dumps(original_body))
    if "contents" not in retry_body:
        retry_body["contents"] = []
    
    last_user_index = -1
    for i in range(len(retry_body["contents"]) - 1, -1, -1):
        if retry_body["contents"][i].get("role") == "user":
            last_user_index = i
            break
            
    history = [
        {"role": "model", "parts": [{"text": accumulated_text}]},
        {"role": "user", "parts": [{"text": "Continue exactly where you left off without any preamble or repetition."}]}
    ]
    
    if last_user_index != -1:
        retry_body["contents"][last_user_index + 1:last_user_index + 1] = history
        logging.debug(f"Inserted retry context after user message at index {last_user_index}")
    else:
        retry_body["contents"].extend(history)
        logging.debug("Appended retry context to end of conversation")
        
    logging.debug(f"Final retry request has {len(retry_body['contents'])} messages")
    return retry_body


async def process_stream_and_retry_internally(
    initial_response: httpx.Response,
    original_request_body: Dict,
    upstream_url: str,
    original_headers: Dict
) -> AsyncGenerator[bytes, None]:
    """The core logic for processing and retrying the SSE stream."""
    accumulated_text = ""
    consecutive_retry_count = 0
    current_response = initial_response
    total_lines_processed = 0
    session_start_time = datetime.now()
    
    is_outputting_formal_text = False
    swallow_mode_active = False

    logging.info(f"Starting stream processing session. Max retries: {CONFIG['max_consecutive_retries']}")

    while True:
        interruption_reason = None
        clean_exit = False
        stream_start_time = datetime.now()
        lines_in_this_stream = 0
        text_in_this_stream = ""

        logging.debug(f"=== Starting stream attempt {consecutive_retry_count + 1}/{CONFIG['max_consecutive_retries'] + 1} ===")
        
        try:
            async for line in sse_line_iterator(current_response):
                total_lines_processed += 1
                lines_in_this_stream += 1

                line_content = parse_line_content(line) if is_data_line(line) else {"text": "", "is_thought": False}
                text_chunk, is_thought = line_content["text"], line_content["is_thought"]

                if swallow_mode_active:
                    if is_thought:
                        logging.debug(f"Swallowing thought chunk due to post-retry filter: {line}")
                        finish_reason_on_swallowed_line = extract_finish_reason(line)
                        if finish_reason_on_swallowed_line:
                            logging.error(f"Stream stopped with reason '{finish_reason_on_swallowed_line}' while swallowing a 'thought' chunk. Triggering retry.")
                            interruption_reason = "FINISH_DURING_THOUGHT"
                            break
                        continue
                    else:
                        logging.info("First formal text chunk received after swallowing. Resuming normal stream.")
                        swallow_mode_active = False

                finish_reason = extract_finish_reason(line)
                needs_retry = False
                
                if finish_reason and is_thought:
                    logging.error(f"Stream stopped with reason '{finish_reason}' on a 'thought' chunk. Triggering retry.")
                    interruption_reason = "FINISH_DURING_THOUGHT"
                    needs_retry = True
                elif is_blocked_line(line):
                    logging.error(f"Content blocked detected in line: {line}")
                    interruption_reason = "BLOCK"
                    needs_retry = True
                elif finish_reason == "STOP":
                    temp_accumulated_text = accumulated_text + text_chunk
                    trimmed_text = temp_accumulated_text.strip()
                    last_char = trimmed_text[-1:]
                    if not (len(trimmed_text) == 0 or last_char in FINAL_PUNCTUATION):
                        logging.error(f"Finish reason 'STOP' treated as incomplete because text ends with '{last_char}'. Triggering retry.")
                        interruption_reason = "FINISH_INCOMPLETE"
                        needs_retry = True
                elif finish_reason and finish_reason not in ("MAX_TOKENS", "STOP"):
                    logging.error(f"Abnormal finish reason: {finish_reason}. Triggering retry.")
                    interruption_reason = "FINISH_ABNORMAL"
                    needs_retry = True

                if needs_retry:
                    break
                
                yield SSE_ENCODER(line + "\n\n")

                if text_chunk and not is_thought:
                    is_outputting_formal_text = True
                    accumulated_text += text_chunk
                    text_in_this_stream += text_chunk

                if finish_reason in ("STOP", "MAX_TOKENS"):
                    logging.info(f"Finish reason '{finish_reason}' accepted as final. Stream complete.")
                    clean_exit = True
                    break
            
            if not clean_exit and interruption_reason is None:
                logging.error("Stream ended without finish reason - detected as DROP")
                interruption_reason = "DROP"

        except httpx.RequestError as e:
            logging.error(f"Exception during stream processing: {e}", exc_info=True)
            interruption_reason = "FETCH_ERROR"
        finally:
            # This generator is responsible for closing the stream it's currently processing.
            if not current_response.is_closed:
                await current_response.aclose()
            
            stream_duration = (datetime.now() - stream_start_time).total_seconds()
            logging.debug("Stream attempt summary:")
            logging.debug(f"  Duration: {stream_duration:.2f}s")
            logging.debug(f"  Lines processed: {lines_in_this_stream}")
            logging.debug(f"  Text generated this stream: {len(text_in_this_stream)} chars")
            logging.debug(f"  Total accumulated text: {len(accumulated_text)} chars")

        if clean_exit:
            session_duration = (datetime.now() - session_start_time).total_seconds()
            logging.info("=== STREAM COMPLETED SUCCESSFULLY ===")
            logging.info(f"Total session duration: {session_duration:.2f}s")
            logging.info(f"Total lines processed: {total_lines_processed}")
            logging.info(f"Total text generated: {len(accumulated_text)} characters")
            logging.info(f"Total retries needed: {consecutive_retry_count}")
            return

        logging.error(f"=== STREAM INTERRUPTED ===\nReason: {interruption_reason}")
        
        if CONFIG["swallow_thoughts_after_retry"] and is_outputting_formal_text:
            logging.info("Retry triggered after formal text output. Will swallow subsequent thought chunks.")
            swallow_mode_active = True

        logging.error(f"Current retry count: {consecutive_retry_count}, Max: {CONFIG['max_consecutive_retries']}")

        if consecutive_retry_count >= CONFIG['max_consecutive_retries']:
            payload = { "error": { "code": 504, "status": "DEADLINE_EXCEEDED", "message": f"Retry limit ({CONFIG['max_consecutive_retries']}) exceeded. Last reason: {interruption_reason}.", "details": [{"@type": "proxy.debug", "accumulated_text_chars": len(accumulated_text)}] } }
            yield SSE_ENCODER(f"event: error\ndata: {json.dumps(payload)}\n\n")
            return

        consecutive_retry_count += 1
        logging.info(f"=== STARTING RETRY {consecutive_retry_count}/{CONFIG['max_consecutive_retries']} ===")

        try:
            retry_body = build_retry_request_body(original_request_body, accumulated_text)
            
            logging.debug(f"Making retry request to: {upstream_url}")
            logging.debug(f"Retry request body size: {len(json.dumps(retry_body))} bytes")

            req = async_client.build_request("POST", upstream_url, headers=original_headers, json=retry_body)
            retry_response = await async_client.send(req, stream=True)
            
            logging.info(f"Retry request completed. Status: {retry_response.status_code} {retry_response.reason_phrase}")

            if retry_response.status_code in NON_RETRYABLE_STATUSES:
                logging.error("=== FATAL ERROR DURING RETRY ===")
                logging.error(f"Received non-retryable status {retry_response.status_code} during retry.")
                error_sse = await write_sse_error_from_upstream(retry_response)
                yield SSE_ENCODER(error_sse)
                await retry_response.aclose()
                return

            retry_response.raise_for_status()

            logging.info(f"✓ Retry attempt {consecutive_retry_count} successful - got new stream")
            current_response = retry_response
        
        except httpx.HTTPStatusError as e:
            logging.error(f"Retry attempt {consecutive_retry_count} failed with status {e.response.status_code}")
            logging.error("This is considered a retryable error - will try again if retries remain")
            if 'retry_response' in locals() and not retry_response.is_closed:
                await e.response.aclose()
            await asyncio.sleep(CONFIG["retry_delay_ms"] / 1000)
        except httpx.RequestError as e:
            logging.error(f"=== RETRY ATTEMPT {consecutive_retry_count} FAILED ===")
            logging.error(f"Exception during retry: {e}", exc_info=True)
            logging.error(f"Will wait {CONFIG['retry_delay_ms']}ms before next attempt")
            await asyncio.sleep(CONFIG["retry_delay_ms"] / 1000)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def handle_request(request: Request, path: str):
    """Main request handler, routing to streaming or non-streaming logic."""
    
    upstream_url = f"{CONFIG['upstream_url_base']}/{path}"
    if request.query_params:
        upstream_url += f"?{request.query_params}"

    logging.info(f"=== NEW REQUEST: {request.method} {request.url} ===")
    logging.info(f"Upstream URL: {upstream_url}")
    
    alt_param = request.query_params.get("alt", "")
    is_stream = "stream" in path or "sse" in alt_param.lower()
    logging.info(f"Detected streaming request: {is_stream}")

    headers_to_forward = {
        k: v for k, v in request.headers.items()
        if k.lower() in ["authorization", "x-goog-api-key", "content-type", "accept"]
    }

    if request.method == "POST" and is_stream:
        try:
            original_request_body = await request.json()
            logging.debug(f"Parsed request body with {len(original_request_body.get('contents', []))} messages")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse request body: {e}")
            return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})

        logging.info("=== MAKING INITIAL STREAMING REQUEST ===")
        try:
            req = async_client.build_request(
                method=request.method,
                url=upstream_url,
                headers=headers_to_forward,
                json=original_request_body,
            )
            initial_response = await async_client.send(req, stream=True)
            
            logging.info(f"Initial response status: {initial_response.status_code} {initial_response.reason_phrase}")
            
            if not initial_response.is_success:
                logging.error("=== INITIAL REQUEST FAILED ===")
                return await standardize_initial_error(initial_response)
            
            stream_generator = process_stream_and_retry_internally(
                initial_response=initial_response,
                original_request_body=original_request_body,
                upstream_url=upstream_url,
                original_headers=headers_to_forward
            )

            return StreamingResponse(
                stream_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )

        except httpx.RequestError as e:
            logging.error(f"Top-level request error for streaming: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"error": "Bad Gateway", "details": str(e)})

    else:
        logging.info("=== MAKING NON-STREAMING REQUEST ===")
        try:
            body = await request.body()
            upstream_req = async_client.build_request(
                method=request.method,
                url=upstream_url,
                headers=headers_to_forward,
                content=body if body else None
            )
            upstream_resp = await async_client.send(upstream_req, stream=True)

            if not upstream_resp.is_success:
                return await standardize_initial_error(upstream_resp)
            
            response_headers = dict(upstream_resp.headers)
            response_headers["Access-Control-Allow-Origin"] = "*"

            return StreamingResponse(
                upstream_resp.aiter_bytes(),
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type"),
                headers=response_headers,
                background=BackgroundTask(upstream_resp.aclose)
            )
        except httpx.RequestError as e:
            logging.error(f"Top-level request error for non-streaming: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"error": "Bad Gateway", "details": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("Starting Gemini API Python Proxy Server...")
    print("URL: http://0.0.0.0:8000")
    print("Upstream Target:", CONFIG["upstream_url_base"])
    print("Press CTRL+C to stop")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

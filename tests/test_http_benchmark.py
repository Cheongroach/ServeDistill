import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import aiohttp
from aiohttp import web
from servedistill.http_benchmark import stream_request, sweep

class HttpStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = "normal"
        self.active = self.peak = 0
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.respond)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        self.session = aiohttp.ClientSession()
        self.row = {"sample_id": "one", "input_sha256": "a", "domain": "instruction",
                    "prompt": "42", "target": {"answer": "42"}}
    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()
    async def respond(self, request):
        payload = await request.json()
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.active += 1
        self.peak = max(self.peak, self.active)
        identity = request.headers["X-Request-Id"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        async def emit(value):
            await response.write(("data: " + json.dumps(value) + "\n\n").encode())
        await emit({"id": identity, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        await asyncio.sleep(0.025)
        await emit({"id": identity if self.mode != "mixed_id" else "other",
                    "choices": [{"index": 0, "delta": {"content": '{"answer":"42"}'},
                                 "finish_reason": "stop"}]})
        await emit({"id": identity, "choices": [],
                    "usage": {"completion_tokens": 7, "prompt_tokens": 12}})
        if self.mode != "truncated":
            await response.write(b"data: [DONE]\n\n")
        self.active -= 1
        await response.write_eof()
        return response
    async def test_first_content_excludes_role_chunk_and_quality_uses_full_text(self):
        record = await stream_request(self.session, self.endpoint, "model", self.row, 128)
        self.assertGreaterEqual(record["ttft_seconds"], 0.02)
        self.assertTrue(record["success"])
        self.assertEqual(record["output_tokens"], 7)
    async def test_stream_without_done_is_rejected(self):
        self.mode = "truncated"
        with self.assertRaisesRegex(RuntimeError, "Incomplete stream"):
            await stream_request(self.session, self.endpoint, "model", self.row, 128)
    async def test_response_ids_cannot_mix_in_one_request(self):
        self.mode = "mixed_id"
        with self.assertRaisesRegex(RuntimeError, "Incomplete stream"):
            await stream_request(self.session, self.endpoint, "model", self.row, 128)
    async def test_clients_are_bounded_and_replenished_without_dropping_rows(self):
        args = SimpleNamespace(endpoint=self.endpoint, model="model", max_new_tokens=128,
                               sweep_timeout=5, gpus=1)
        rows = [{**self.row, "sample_id": str(i)} for i in range(9)]
        with tempfile.TemporaryDirectory() as folder:
            result = await sweep(self.session, args, rows, 3, Path(folder))
            with (Path(folder)/"requests.jsonl").open() as stream:
                evidence = [json.loads(line) for line in stream]
        self.assertEqual(self.peak, 3)
        self.assertEqual(result["n"], 9)
        self.assertEqual({r["sample_id"] for r in evidence}, {str(i) for i in range(9)})
        self.assertEqual(result["success_fraction"], 1)

if __name__ == "__main__": unittest.main()

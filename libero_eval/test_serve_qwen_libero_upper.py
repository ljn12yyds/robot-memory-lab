"""Exercise the bounded Qwen HTTP routes without loading model weights."""

import json
from io import BytesIO
import unittest

from serve_qwen_libero_upper import make_handler


class QwenMemoryRoutesTest(unittest.TestCase):
    def test_cue_and_memory_requests_are_separate(self):
        self.calls = []

        def fake_generator(system, prompt, image_path):
            self.calls.append((system, prompt, image_path))
            if "one-time visual task cue" in system:
                return '{"cue":"GREEN"}'
            return '{"next":"CONTINUE"}'

        handler_class = make_handler(fake_generator)

        def post(route, payload):
            request_bytes = json.dumps(payload).encode("utf-8")
            handler = handler_class.__new__(handler_class)
            handler.path = route
            handler.headers = {"Content-Length": str(len(request_bytes))}
            handler.rfile = BytesIO(request_bytes)
            handler.wfile = BytesIO()
            handler.send_response = lambda code: self.assertEqual(code, 200)
            handler.send_header = lambda *args: None
            handler.end_headers = lambda: None
            handler.do_POST()
            return json.loads(handler.wfile.getvalue())

        cue = post("/read_cue", {"image_path": "initial.jpg"})
        next_step = post(
            "/next_from_memory",
            {
                "task_memory": {"current_step": 1, "world_state": {"ketchup_1": "in_basket"}},
                "image_path": "current.jpg",
            },
        )

        self.assertEqual(cue["cue"], "GREEN")
        self.assertEqual(next_step["next"], "CONTINUE")
        self.assertEqual(self.calls[0][2], "initial.jpg")
        self.assertEqual(self.calls[1][2], "current.jpg")
        self.assertEqual(
            json.loads(self.calls[1][1])["task_memory"],
            {"current_step": 1, "world_state": {"ketchup_1": "in_basket"}},
        )


if __name__ == "__main__":
    unittest.main()

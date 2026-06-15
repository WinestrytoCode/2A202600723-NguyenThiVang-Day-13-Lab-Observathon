"""Test case to verify wrapper.py mitigate function behaves correctly under mock conditions.
Ensures no exceptions/wrapper_errors are raised by our tracing/mitigation layer.
"""
import unittest
import os
from unittest.mock import MagicMock
from solution.wrapper import mitigate


class TestWrapperMitigation(unittest.TestCase):
    def setUp(self):
        # Set dummy env vars for Langfuse if they are not present, to ensure factory doesn't crash
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-test"
        os.environ["OBS_BACKEND"] = "file"  # use file backend for offline tests

    def test_mitigate_passthrough(self):
        def mock_call_next(question, config):
            return {
                "answer": "Tong cong: 100000 VND",
                "status": "ok",
                "steps": 1,
                "trace": [
                    {"action": "check_stock", "args": {"product": "ipad"}, "observation": "in_stock=True"}
                ],
                "meta": {
                    "latency_ms": 150,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    "model": "gpt-5.4-nano"
                }
            }

        context = {
            "session_id": "test-session-123",
            "turn_index": 0,
            "qid": "test-qid-123",
            "cache": {},
            "cache_lock": MagicMock()
        }
        config = {
            "temperature": 0.2
        }

        try:
            res = mitigate(mock_call_next, "Mua 1 ipad", config, context)
            self.assertEqual(res["status"], "ok")
            self.assertIn("Tong cong", res["answer"])
        except Exception as e:
            self.fail(f"mitigate raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from agent.protocol import try_parse_tool_call


class ProtocolToolCallParsingTests(unittest.TestCase):
    def test_strict_tool_call_json_parses(self) -> None:
        raw = '{"type":"tool_call","name":"read_text_file","args":{"path":"a.md"}}'
        parsed = try_parse_tool_call(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.tool_call.name, "read_text_file")
        self.assertEqual(parsed.tool_call.args, {"path": "a.md"})
        self.assertEqual(parsed.trailing_text, "")

    def test_prefix_tool_call_json_at_start_parses(self) -> None:
        raw = ' \n{"type":"tool_call","name":"read_text_file","args":{"path":"a.md"}} trailing'
        parsed = try_parse_tool_call(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.tool_call.name, "read_text_file")
        self.assertEqual(parsed.tool_call.args, {"path": "a.md"})
        self.assertEqual(parsed.trailing_text, "trailing")

    def test_rejects_non_whitespace_prefix_before_json(self) -> None:
        raw = 'preface {"type":"tool_call","name":"read_text_file","args":{"path":"a.md"}} trailing'
        parsed = try_parse_tool_call(raw)
        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()

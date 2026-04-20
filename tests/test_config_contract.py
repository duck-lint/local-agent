from __future__ import annotations

import unittest

from agent.config import build_app_config, deep_merge_config, DEFAULT_CONFIG


class ConfigContractTests(unittest.TestCase):
    def test_build_app_config_rejects_obsolete_pipeline_keys(self) -> None:
        obsolete_key = "".join(["ph", "ase2"])
        with self.assertRaises(ValueError):
            build_app_config({obsolete_key: {"index_db_path": "index/index.sqlite"}})

    def test_build_app_config_reads_new_vocabulary(self) -> None:
        cfg = build_app_config(
            deep_merge_config(
                DEFAULT_CONFIG,
                {
                    "corpus": {"db_path": "index/custom.sqlite", "max_chars": 256},
                    "grounding": {"evidence_top_n": 4},
                    "memory": {"db_path": "memory/custom.sqlite"},
                },
            )
        )
        self.assertEqual(cfg.corpus.db_path, "index/custom.sqlite")
        self.assertEqual(cfg.corpus.max_chars, 256)
        self.assertEqual(cfg.grounding.evidence_top_n, 4)
        self.assertEqual(cfg.memory.db_path, "memory/custom.sqlite")


if __name__ == "__main__":
    unittest.main()

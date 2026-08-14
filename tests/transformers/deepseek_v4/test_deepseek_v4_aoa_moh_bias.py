# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for ``indexer_moh_bias`` save/load round-trip.

``indexer_moh_bias`` is a *persistable* buffer that the aux-loss-free callback
mutates every optimizer step. The HF↔Fleet AOA config must therefore export it
in *both* directions:

  * HF -> Fleet (``_gen_aoa_config``): zero-init on load, since a fresh HF
    checkpoint doesn't carry it. Uses the ``_ -> ...indexer_moh_bias`` add
    primitive.

  * Fleet -> HF (``_gen_inv_aoa_config``): persist the trained bias back into
    the HF checkpoint on ``save_pretrained``. Without this side of the pair,
    every save/load round-trip resets the load-balancing state to zero and
    silently loses training progress.

These tests parse the AOA statement lists produced by both classmethods and
assert the symmetry directly, so any future refactor that drops one side
fails here.
"""

import re
import unittest

from paddleformers.transformers.deepseek_v4.configuration import DeepseekV4Config
from paddleformers.transformers.deepseek_v4.modeling import DeepseekV4PreTrainedModel

# ---------------------------------------------------------------------------
# Config factory tuned to build both a decoder-side and an MTP-side CSAIndexer,
# so both indexer branches of the AOA config are exercised.
# ---------------------------------------------------------------------------


def _moh_config(**overrides):
    """A minimal DSv4 config with MoH ON and both branches populated.

    * ``csa_compress_ratios[0] = 4`` -> layer 0 has a CSAIndexer (decoder branch).
    * ``mtp_num_layers = 1`` and the MTP slot's compress ratio is ``4``
      -> the MTP branch also has a CSAIndexer.
    """
    kwargs = dict(
        num_hidden_layers=1,
        n_routed_experts=2,
        # csa_compress_ratios has length = num_hidden_layers + mtp_num_layers.
        # index 0 is the decoder layer; the tail is consumed by MTP indexer
        # branch (only compress_ratio > 0 and <= 4 builds an indexer).
        csa_compress_ratios=[4, 4],
        csa_dense_mode=False,
        mtp_num_layers=1,
        use_moh=True,
        num_activated_heads=8,
        dsa_index_n_heads=64,
        # Keep expert count trivial so per-expert AOA lines don't drown out
        # the indexer lines we're asserting on.
        moe_n_hash_layers=0,
    )
    kwargs.update(overrides)
    return DeepseekV4Config(**kwargs)


def _find_indexer_lines(stmts, direction):
    """Return the subset of AOA statements that touch ``indexer_moh_bias``.

    ``direction`` is 'fwd' (HF -> Fleet, expect ``_`` on the LHS) or 'inv'
    (Fleet -> HF, expect the Fleet name on the LHS and HF on the RHS).
    """
    out = []
    for s in stmts:
        if "indexer_moh_bias" not in s:
            continue
        out.append(s.strip())
    return out


def _aoa_stmts(config, direction):
    """Extract the statement list from the ``_gen_[inv_]aoa_config`` return.

    Both methods return ``{"aoa_statements": [...]}`` (the dict form is what
    ``PaddleFormers`` hands to ``AoAExecutor``); we only care about the flat
    statement strings for these regressions.
    """
    fn = (
        DeepseekV4PreTrainedModel._gen_aoa_config
        if direction == "fwd"
        else DeepseekV4PreTrainedModel._gen_inv_aoa_config
    )
    out = fn(config)
    if isinstance(out, dict):
        return out["aoa_statements"]
    # Historical form: a flat list.
    return list(out)


class TestIndexerMoHBiasRoundTrip(unittest.TestCase):
    """Both AOA directions must carry ``indexer_moh_bias``."""

    def test_forward_aoa_zero_inits_bias(self):
        """HF -> Fleet: ``_ -> ....indexer_moh_bias`` on both decoder & MTP."""
        cfg = _moh_config()
        stmts = _aoa_stmts(cfg, "fwd")
        bias_lines = _find_indexer_lines(stmts, "fwd")
        # One line for the decoder indexer, one for the MTP indexer.
        self.assertEqual(
            len(bias_lines),
            2,
            f"expected 2 indexer_moh_bias entries in HF->Fleet AOA, got {bias_lines}",
        )
        for line in bias_lines:
            # LHS must be the add-primitive '_'.
            self.assertRegex(line, r"^_\s*->\s*.*indexer_moh_bias\b")

    def test_inverse_aoa_persists_bias(self):
        """Fleet -> HF: named -> named mapping on both decoder & MTP.

        Regression guard against the bug where ``indexer_moh_bias`` was
        missing from ``_gen_inv_aoa_config``, so every ``save_pretrained``
        silently dropped the trained aux-loss-free bias.
        """
        cfg = _moh_config()
        stmts = _aoa_stmts(cfg, "inv")
        bias_lines = _find_indexer_lines(stmts, "inv")
        self.assertEqual(
            len(bias_lines),
            2,
            f"expected 2 indexer_moh_bias entries in Fleet->HF AOA, got {bias_lines}",
        )
        pattern = re.compile(r"^([\w\.]+)\.indexer_moh_bias\s*->\s*([\w\.]+)\.indexer_moh_bias\b")
        for line in bias_lines:
            m = pattern.match(line)
            self.assertIsNotNone(
                m,
                f"inverse AOA line must be a named->named mapping, got: {line!r}",
            )
            fleet_prefix, hf_prefix = m.group(1), m.group(2)
            # Fleet side: ``...self_attn.core_attention.indexer``.
            self.assertIn("self_attn.core_attention.indexer", fleet_prefix)
            # HF side: ``...attn.indexer`` (decoder or MTP), NOT the Fleet form.
            self.assertTrue(
                hf_prefix.endswith("attn.indexer"),
                f"unexpected HF prefix for indexer_moh_bias: {hf_prefix!r}",
            )
            self.assertNotIn("core_attention", hf_prefix)

    def test_round_trip_pairs_line_up(self):
        """Every HF->Fleet target must have a matching Fleet->HF source.

        This is the direct assertion the reviewer asked for: for each
        ``_ -> X.indexer_moh_bias`` in the forward config, there must be a
        corresponding ``X.indexer_moh_bias -> _`` (any HF target) in the
        inverse config with the *same* Fleet path.
        """
        cfg = _moh_config()
        fwd = _aoa_stmts(cfg, "fwd")
        inv = _aoa_stmts(cfg, "inv")

        # Extract Fleet-side prefixes touched by the forward add-primitive.
        fwd_prefixes = set()
        fwd_re = re.compile(r"^_\s*->\s*([\w\.]+)\.indexer_moh_bias")
        for s in fwd:
            m = fwd_re.match(s.strip())
            if m:
                fwd_prefixes.add(m.group(1))

        # Extract Fleet-side prefixes on the LHS of the inverse mapping.
        inv_prefixes = set()
        inv_re = re.compile(r"^([\w\.]+)\.indexer_moh_bias\s*->")
        for s in inv:
            m = inv_re.match(s.strip())
            if m:
                inv_prefixes.add(m.group(1))

        self.assertEqual(
            fwd_prefixes,
            inv_prefixes,
            "HF->Fleet and Fleet->HF must cover the same set of "
            "indexer_moh_bias sites; drift here means save_pretrained will "
            "silently drop the trained bias.",
        )
        self.assertGreater(
            len(fwd_prefixes),
            0,
            "sanity: the test config should exercise at least one indexer",
        )


class TestIndexerMoHBiasOnlyWhenEnabled(unittest.TestCase):
    """No ``indexer_moh_bias`` lines when ``use_moh=False`` (both directions)."""

    def test_forward_no_bias_when_moh_disabled(self):
        cfg = _moh_config(use_moh=False, num_activated_heads=None)
        stmts = _aoa_stmts(cfg, "fwd")
        self.assertEqual(_find_indexer_lines(stmts, "fwd"), [])

    def test_inverse_no_bias_when_moh_disabled(self):
        cfg = _moh_config(use_moh=False, num_activated_heads=None)
        stmts = _aoa_stmts(cfg, "inv")
        self.assertEqual(_find_indexer_lines(stmts, "inv"), [])


if __name__ == "__main__":
    unittest.main()

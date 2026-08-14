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


def _aoa_stmts(config, direction, checkpoint_keys=None):
    """Extract the statement list from the ``_gen_[inv_]aoa_config`` return.

    Both methods return ``{"aoa_statements": [...]}`` (the dict form is what
    ``PaddleFormers`` hands to ``AoAExecutor``); we only care about the flat
    statement strings for these regressions.

    ``checkpoint_keys`` is forwarded to ``_gen_aoa_config`` (fwd only) so the
    tests can exercise the "HF checkpoint carries the trained bias" branch;
    ``None`` preserves the historical zero-init fallback. Ignored for the
    inverse direction, which has no dependency on the loaded key set.
    """
    fn = (
        DeepseekV4PreTrainedModel._gen_aoa_config
        if direction == "fwd"
        else DeepseekV4PreTrainedModel._gen_inv_aoa_config
    )
    if direction == "fwd":
        try:
            out = fn(config, checkpoint_keys=checkpoint_keys)
        except TypeError:
            # Historical classmethod without the kwarg -- covered by the
            # ``None`` case anyway; re-raise if a real key set was passed
            # since that means the code hasn't picked up the kwarg yet.
            if checkpoint_keys is not None:
                raise
            out = fn(config)
    else:
        out = fn(config)
    if isinstance(out, dict):
        return out["aoa_statements"]
    # Historical form: a flat list.
    return list(out)


class TestIndexerMoHBiasRoundTrip(unittest.TestCase):
    """Both AOA directions must carry ``indexer_moh_bias``."""

    def test_forward_aoa_zero_inits_bias(self):
        """HF -> Fleet: ``_ -> ....indexer_moh_bias`` on both decoder & MTP.

        This is the *fresh HF release* / legacy caller path (no
        ``checkpoint_keys`` info), so the forward AOA must still fall back
        to the add primitive so the buffer is zero-initialized.
        """
        cfg = _moh_config()
        stmts = _aoa_stmts(cfg, "fwd")  # checkpoint_keys=None (legacy path)
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

    def test_forward_aoa_loads_bias_when_checkpoint_carries_it(self):
        """HF -> Fleet: ``hf_key -> fleet_key`` when the HF checkpoint has the bias.

        This is the *round-trip load* path: the previous ``save_pretrained``
        wrote ``layers.*.attn.indexer.indexer_moh_bias`` via
        ``_gen_inv_aoa_config``, so the next ``from_pretrained`` must load
        that trained state instead of overwriting it with zeros. Regression
        guard for the P1 the reviewer flagged.
        """
        cfg = _moh_config()

        # Enumerate the HF-side prefixes the inverse config exports (this is
        # the exact key set the round-tripped checkpoint will contain).
        inv_stmts = _aoa_stmts(cfg, "inv")
        inv_re = re.compile(r"^[\w\.]+\.indexer_moh_bias\s*->\s*([\w\.]+\.indexer_moh_bias)\b")
        hf_bias_keys = set()
        for s in inv_stmts:
            m = inv_re.match(s.strip())
            if m:
                hf_bias_keys.add(m.group(1))
        self.assertGreater(
            len(hf_bias_keys),
            0,
            "test setup: inverse AOA must export at least one indexer_moh_bias key",
        )

        stmts = _aoa_stmts(cfg, "fwd", checkpoint_keys=hf_bias_keys)
        bias_lines = _find_indexer_lines(stmts, "fwd")
        self.assertEqual(
            len(bias_lines),
            2,
            f"expected 2 indexer_moh_bias entries in HF->Fleet AOA, got {bias_lines}",
        )
        pattern = re.compile(r"^([\w\.]+)\.indexer_moh_bias\s*->\s*([\w\.]+)\.indexer_moh_bias\b")
        for line in bias_lines:
            m = pattern.match(line)
            self.assertIsNotNone(
                m,
                f"forward AOA with checkpoint_keys must be a named->named " f"mapping (not '_ -> ...'), got: {line!r}",
            )
            hf_prefix, fleet_prefix = m.group(1), m.group(2)
            # HF side (LHS): ``layers.*.attn.indexer`` (decoder or MTP).
            self.assertTrue(
                hf_prefix.endswith("attn.indexer"),
                f"unexpected HF LHS for indexer_moh_bias: {hf_prefix!r}",
            )
            # Fleet side (RHS): ``...self_attn.core_attention.indexer``.
            self.assertIn("self_attn.core_attention.indexer", fleet_prefix)
            # The '_' add primitive must NOT appear when the key is present.
            self.assertFalse(
                line.strip().startswith("_"),
                f"AOA still zero-inits a bias that IS in the checkpoint: {line!r}",
            )

    def test_forward_aoa_mixed_checkpoint(self):
        """Partial round-trip: only some indexer sites carry the trained bias.

        E.g. a checkpoint saved by an older Fleet where only the decoder
        branch had ``indexer_moh_bias`` -- the MTP branch is still fresh.
        The forward AOA must emit named->named for the present key AND
        ``_ -> ...`` for the missing one, not one rule for both.
        """
        cfg = _moh_config()
        inv_stmts = _aoa_stmts(cfg, "inv")
        inv_re = re.compile(r"^[\w\.]+\.indexer_moh_bias\s*->\s*([\w\.]+\.indexer_moh_bias)\b")
        all_hf_keys = [inv_re.match(s.strip()).group(1) for s in inv_stmts if inv_re.match(s.strip())]
        self.assertEqual(len(all_hf_keys), 2, "test setup: need exactly 2 HF bias keys")
        # Keep only the decoder-side key (the one that does NOT contain
        # ``transformer_layer`` on the Fleet side -- but we're keying by HF
        # names here, so filter by MTP prefix instead).
        mtp_key = next((k for k in all_hf_keys if "mtp" in k or k.startswith("mtp")), None)
        # Fallback: HF-side MTP layers live at ``layers.{num_decoder+i}.attn.indexer``
        # in this codebase, so treat the second key as MTP if the first isn't.
        if mtp_key is None:
            mtp_key = all_hf_keys[1]
        decoder_key = next(k for k in all_hf_keys if k != mtp_key)

        # Only the decoder-side bias is present in this "partial" checkpoint.
        stmts = _aoa_stmts(cfg, "fwd", checkpoint_keys={decoder_key})
        bias_lines = _find_indexer_lines(stmts, "fwd")
        self.assertEqual(len(bias_lines), 2, f"got {bias_lines}")

        named_lines = [ln for ln in bias_lines if not ln.strip().startswith("_")]
        add_lines = [ln for ln in bias_lines if ln.strip().startswith("_")]
        self.assertEqual(
            len(named_lines),
            1,
            f"exactly one named->named line expected for the present key, got {named_lines}",
        )
        self.assertEqual(
            len(add_lines),
            1,
            f"exactly one '_ -> ...' line expected for the missing key, got {add_lines}",
        )
        # The named line must use the decoder key on the LHS.
        self.assertTrue(
            named_lines[0].strip().startswith(decoder_key),
            f"named line should route from {decoder_key!r}, got: {named_lines[0]!r}",
        )

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

    def test_save_load_round_trip_end_to_end(self):
        """Full save-then-load: no site is zero-init'd after a round-trip.

        Directly models the reviewer's P1: the previous fix only closed the
        Fleet -> HF export side; a subsequent ``from_pretrained`` of *that*
        checkpoint had no way to know it should load the exported bias
        instead of zeroing it back out via the ``_ -> ...`` add primitive.

        This test simulates the save/load pair:
          1. Ask the inverse config which HF keys ``save_pretrained`` would
             write for the bias (build ``checkpoint_keys``).
          2. Ask the forward config what it would emit given exactly that
             key set (i.e. the ``from_pretrained`` right after save).
          3. Assert every emitted bias line for a site the checkpoint DID
             persist is a named->named mapping -- NEVER ``_ -> ...``.

        Failure mode this catches: bias is written by save_pretrained but
        the load path still uses the add primitive, so ``from_pretrained``
        of a fresh save silently resets the trained load-balancing state.
        """
        cfg = _moh_config()
        inv_stmts = _aoa_stmts(cfg, "inv")
        # Set of HF-side keys that save_pretrained will actually persist.
        inv_re = re.compile(r"^[\w\.]+\.indexer_moh_bias\s*->\s*([\w\.]+\.indexer_moh_bias)\b")
        persisted_hf_keys = set()
        for s in inv_stmts:
            m = inv_re.match(s.strip())
            if m:
                persisted_hf_keys.add(m.group(1))
        self.assertGreater(len(persisted_hf_keys), 0)

        # Simulate ``from_pretrained`` right after save -- pass those keys in.
        fwd_stmts = _aoa_stmts(cfg, "fwd", checkpoint_keys=persisted_hf_keys)
        fwd_bias_lines = _find_indexer_lines(fwd_stmts, "fwd")
        self.assertEqual(
            len(fwd_bias_lines),
            len(persisted_hf_keys),
            f"expected one forward bias line per persisted site, "
            f"got {fwd_bias_lines} for keys {persisted_hf_keys}",
        )

        # None of them may be the zero-init add primitive.
        offenders = [ln for ln in fwd_bias_lines if ln.strip().startswith("_")]
        self.assertEqual(
            offenders,
            [],
            "round-trip broken: save_pretrained persisted these HF keys "
            f"{persisted_hf_keys}, but the forward AOA still zero-inits at "
            f"least one of them: {offenders}. This is the exact regression "
            "the reviewer flagged -- trained aux-loss-free bias is lost on "
            "the next load.",
        )

        # Also verify every LHS is actually one of the persisted HF keys
        # (not some fabricated name that doesn't line up with the save side).
        pattern = re.compile(r"^([\w\.]+\.indexer_moh_bias)\s*->")
        emitted_lhs = set()
        for line in fwd_bias_lines:
            m = pattern.match(line.strip())
            self.assertIsNotNone(m, f"malformed forward bias line: {line!r}")
            emitted_lhs.add(m.group(1))
        self.assertEqual(
            emitted_lhs,
            persisted_hf_keys,
            "forward AOA reads a different HF key set than the inverse "
            "AOA writes; save/load will silently mismatch.",
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

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

import unittest
import warnings

import numpy as np


class TestAudioUtils(unittest.TestCase):
    """Tests for paddleformers.transformers.audio_utils module."""

    def test_hertz_to_mel_htk_scalar(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        self.assertAlmostEqual(hertz_to_mel(0.0, "htk"), 0.0, places=5)

        result = hertz_to_mel(1000.0, "htk")
        expected = 2595.0 * np.log10(1.0 + 1000.0 / 700.0)
        self.assertAlmostEqual(result, expected, places=5)

    def test_hertz_to_mel_htk_array(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        freqs = np.array([0.0, 1000.0, 4000.0])
        mels = hertz_to_mel(freqs, "htk")
        self.assertEqual(mels.shape, (3,))
        self.assertAlmostEqual(mels[0], 0.0, places=5)
        self.assertAlmostEqual(mels[1], 2595.0 * np.log10(1.0 + 1000.0 / 700.0), places=5)

    def test_hertz_to_mel_slaney_scalar_low(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        result = hertz_to_mel(500.0, "slaney")
        expected = 3.0 * 500.0 / 200.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_hertz_to_mel_slaney_scalar_high(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        result = hertz_to_mel(2000.0, "slaney")
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        expected = min_log_mel + np.log(2000.0 / min_log_hertz) * logstep
        self.assertAlmostEqual(result, expected, places=5)

    def test_hertz_to_mel_slaney_boundary(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        result = hertz_to_mel(1000.0, "slaney")
        self.assertAlmostEqual(result, 15.0, places=5)

    def test_hertz_to_mel_slaney_array_mixed(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        freqs = np.array([200.0, 500.0, 1000.0, 2000.0, 4000.0])
        mels = hertz_to_mel(freqs, "slaney")
        self.assertEqual(mels.shape, (5,))
        np.testing.assert_allclose(mels[0], 3.0 * 200.0 / 200.0, rtol=1e-6)
        np.testing.assert_allclose(mels[1], 3.0 * 500.0 / 200.0, rtol=1e-6)
        self.assertAlmostEqual(mels[2], 15.0, places=5)

    def test_hertz_to_mel_invalid_scale(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        with self.assertRaises(ValueError):
            hertz_to_mel(1000.0, "invalid_scale")

    def test_hertz_to_mel_default_scale_is_htk(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel

        result_default = hertz_to_mel(1000.0)
        result_explicit = hertz_to_mel(1000.0, "htk")
        self.assertAlmostEqual(result_default, result_explicit, places=10)

    def test_mel_to_hertz_htk_scalar(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        self.assertAlmostEqual(mel_to_hertz(0.0, "htk"), 0.0, places=5)

        original_freq = 1000.0
        mel_val = 2595.0 * np.log10(1.0 + original_freq / 700.0)
        recovered = mel_to_hertz(mel_val, "htk")
        self.assertAlmostEqual(recovered, original_freq, places=4)

    def test_mel_to_hertz_htk_array(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        mels = np.array([0.0, 1000.0, 2000.0])
        freqs = mel_to_hertz(mels, "htk")
        self.assertEqual(freqs.shape, (3,))

    def test_mel_to_hertz_slaney_scalar_low(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        result = mel_to_hertz(7.5, "slaney")
        expected = 200.0 * 7.5 / 3.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_mel_to_hertz_slaney_scalar_high(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        result = mel_to_hertz(20.0, "slaney")
        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        expected = min_log_hertz * np.exp(logstep * (20.0 - min_log_mel))
        self.assertAlmostEqual(result, expected, places=5)

    def test_mel_to_hertz_slaney_boundary(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        result = mel_to_hertz(15.0, "slaney")
        self.assertAlmostEqual(result, 1000.0, places=5)

    def test_mel_to_hertz_slaney_array_mixed(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        mels = np.array([3.0, 7.5, 15.0, 20.0, 30.0])
        freqs = mel_to_hertz(mels, "slaney")
        self.assertEqual(freqs.shape, (5,))

    def test_mel_to_hertz_invalid_scale(self):
        from paddleformers.transformers.audio_utils import mel_to_hertz

        with self.assertRaises(ValueError):
            mel_to_hertz(1000.0, "invalid")

    def test_hertz_mel_roundtrip_htk(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel, mel_to_hertz

        freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0])
        mels = hertz_to_mel(freqs, "htk")
        recovered = mel_to_hertz(mels, "htk")
        np.testing.assert_allclose(freqs, recovered, rtol=1e-5)

    def test_hertz_mel_roundtrip_slaney(self):
        from paddleformers.transformers.audio_utils import hertz_to_mel, mel_to_hertz

        freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0])
        mels = hertz_to_mel(freqs, "slaney")
        recovered = mel_to_hertz(mels, "slaney")
        np.testing.assert_allclose(freqs, recovered, rtol=1e-5)

    def test_create_triangular_filter_bank(self):
        from paddleformers.transformers.audio_utils import (
            _create_triangular_filter_bank,
        )

        fft_freqs = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        filter_freqs = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        self.assertEqual(filters.shape, (6, 4))

    def test_create_triangular_filter_bank_shape(self):
        from paddleformers.transformers.audio_utils import (
            _create_triangular_filter_bank,
        )

        num_freq_bins = 257
        num_mel_filters = 80
        fft_freqs = np.linspace(0, 8000, num_freq_bins)
        filter_freqs = np.linspace(0, 8000, num_mel_filters + 2)
        filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        self.assertEqual(filters.shape, (num_freq_bins, num_mel_filters))

    def test_create_triangular_filter_bank_non_negative(self):
        from paddleformers.transformers.audio_utils import (
            _create_triangular_filter_bank,
        )

        fft_freqs = np.linspace(0, 4000, 257)
        filter_freqs = np.linspace(0, 4000, 42)
        fb = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        self.assertTrue(np.all(fb >= 0))

    def test_mel_filter_bank_basic(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        filters = mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=80,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
        )
        self.assertEqual(filters.shape, (257, 80))

    def test_mel_filter_bank_slaney_norm(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        filters = mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=80,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
            norm="slaney",
        )
        self.assertEqual(filters.shape, (257, 80))

    def test_mel_filter_bank_no_norm(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        fb = mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=40,
            min_frequency=80.0,
            max_frequency=4000.0,
            sampling_rate=8000,
            norm=None,
            mel_scale="htk",
        )
        self.assertEqual(fb.shape, (257, 40))

    def test_mel_filter_bank_invalid_norm(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        with self.assertRaises(ValueError):
            mel_filter_bank(
                num_frequency_bins=257,
                num_mel_filters=80,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
                norm="invalid",
            )

    def test_mel_filter_bank_too_many_filters_warning(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mel_filter_bank(
                num_frequency_bins=5,
                num_mel_filters=100,
                min_frequency=0.0,
                max_frequency=8000.0,
                sampling_rate=16000,
            )
            self.assertTrue(len(w) > 0)
            self.assertTrue(any("mel filter has all zero values" in str(warning.message) for warning in w))

    def test_mel_filter_bank_slaney_norm_finite_values(self):
        from paddleformers.transformers.audio_utils import mel_filter_bank

        fb = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=80,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.assertTrue(np.isfinite(fb).all())

    def test_optimal_fft_length_power_of_two(self):
        from paddleformers.transformers.audio_utils import optimal_fft_length

        self.assertEqual(optimal_fft_length(256), 256)
        self.assertEqual(optimal_fft_length(512), 512)

    def test_optimal_fft_length_non_power_of_two(self):
        from paddleformers.transformers.audio_utils import optimal_fft_length

        self.assertEqual(optimal_fft_length(400), 512)
        self.assertEqual(optimal_fft_length(100), 128)
        self.assertEqual(optimal_fft_length(300), 512)

    def test_optimal_fft_length_small(self):
        from paddleformers.transformers.audio_utils import optimal_fft_length

        self.assertEqual(optimal_fft_length(1), 1)
        self.assertEqual(optimal_fft_length(3), 4)

    def test_window_function_hann_periodic(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="hann", periodic=True)
        self.assertEqual(win.shape, (400,))
        self.assertTrue(np.all(win >= 0.0))
        self.assertTrue(np.all(win <= 1.0))

    def test_window_function_hann_symmetric(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="hann", periodic=False)
        self.assertEqual(win.shape, (400,))

    def test_window_function_hamming(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="hamming", periodic=True)
        self.assertEqual(win.shape, (400,))

    def test_window_function_hamming_window_alias(self):
        from paddleformers.transformers.audio_utils import window_function

        w1 = window_function(100, name="hamming", periodic=True)
        w2 = window_function(100, name="hamming_window", periodic=True)
        np.testing.assert_allclose(w1, w2, rtol=1e-6)

    def test_window_function_hann_window_alias(self):
        from paddleformers.transformers.audio_utils import window_function

        w1 = window_function(100, name="hann", periodic=True)
        w2 = window_function(100, name="hann_window", periodic=True)
        np.testing.assert_allclose(w1, w2, rtol=1e-6)

    def test_window_function_boxcar(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="boxcar", periodic=True)
        self.assertEqual(win.shape, (400,))
        np.testing.assert_array_almost_equal(win, np.ones(400))

    def test_window_function_invalid_name(self):
        from paddleformers.transformers.audio_utils import window_function

        with self.assertRaises(ValueError):
            window_function(400, name="invalid_window")

    def test_window_function_with_frame_length_centered(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="hann", periodic=True, frame_length=512, center=True)
        self.assertEqual(win.shape, (512,))

    def test_window_function_with_frame_length_no_center(self):
        from paddleformers.transformers.audio_utils import window_function

        win = window_function(400, name="hann", periodic=True, frame_length=512, center=False)
        self.assertEqual(win.shape, (512,))
        self.assertTrue(np.all(win[400:] == 0.0))

    def test_window_function_frame_length_too_small(self):
        from paddleformers.transformers.audio_utils import window_function

        with self.assertRaises(ValueError):
            window_function(400, name="hann", frame_length=200)

    def test_power_to_db_basic(self):
        from paddleformers.transformers.audio_utils import power_to_db

        spectrogram = np.array([[1.0, 2.0], [3.0, 4.0]])
        db = power_to_db(spectrogram, reference=1.0)
        expected = 10.0 * np.log10(spectrogram)
        np.testing.assert_array_almost_equal(db, expected)

    def test_power_to_db_with_reference(self):
        from paddleformers.transformers.audio_utils import power_to_db

        spectrogram = np.array([[10.0, 20.0], [30.0, 40.0]])
        db = power_to_db(spectrogram, reference=10.0)
        expected = 10.0 * (np.log10(spectrogram) - np.log10(10.0))
        np.testing.assert_array_almost_equal(db, expected)

    def test_power_to_db_reference_less_than_min(self):
        from paddleformers.transformers.audio_utils import power_to_db

        spectrogram = np.array([[1.0, 2.0]])
        db = power_to_db(spectrogram, reference=0.01, min_value=0.1)
        self.assertTrue(np.all(np.isfinite(db)))

    def test_power_to_db_with_db_range(self):
        from paddleformers.transformers.audio_utils import power_to_db

        spectrogram = np.array([[1e-5, 1e-3], [1.0, 10.0]])
        db = power_to_db(spectrogram, reference=1.0, db_range=80.0)
        self.assertTrue(db.max() - db.min() <= 80.0 + 1e-5)

    def test_power_to_db_no_db_range(self):
        from paddleformers.transformers.audio_utils import power_to_db

        spectrogram = np.array([[1.0, 100.0]])
        db = power_to_db(spectrogram, reference=100.0)
        self.assertTrue(np.all(np.isfinite(db)))

    def test_power_to_db_invalid_reference(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), reference=0.0)

    def test_power_to_db_negative_reference(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), reference=-1.0)

    def test_power_to_db_invalid_min_value(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), min_value=0.0)

    def test_power_to_db_negative_min_value(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), min_value=-0.5)

    def test_power_to_db_invalid_db_range(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), db_range=-10.0)

    def test_power_to_db_zero_db_range(self):
        from paddleformers.transformers.audio_utils import power_to_db

        with self.assertRaises(ValueError):
            power_to_db(np.array([[1.0]]), db_range=0.0)

    def test_amplitude_to_db_basic(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        spectrogram = np.array([[1.0, 2.0], [3.0, 4.0]])
        db = amplitude_to_db(spectrogram, reference=1.0)
        expected = 20.0 * np.log10(spectrogram)
        np.testing.assert_array_almost_equal(db, expected)

    def test_amplitude_to_db_with_reference(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        spectrogram = np.array([[10.0, 20.0], [30.0, 40.0]])
        db = amplitude_to_db(spectrogram, reference=10.0)
        expected = 20.0 * (np.log10(spectrogram) - np.log10(10.0))
        np.testing.assert_array_almost_equal(db, expected)

    def test_amplitude_to_db_with_db_range(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        spectrogram = np.array([[1e-3, 1e-2], [0.1, 1.0]])
        db = amplitude_to_db(spectrogram, reference=1.0, db_range=80.0)
        self.assertTrue(db.max() - db.min() <= 80.0 + 1e-5)

    def test_amplitude_to_db_reference_less_than_min(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        spectrogram = np.array([[1.0, 2.0]])
        db = amplitude_to_db(spectrogram, reference=0.001, min_value=0.01)
        self.assertTrue(np.all(np.isfinite(db)))

    def test_amplitude_to_db_no_db_range(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        spectrogram = np.array([[1.0, 100.0]])
        db = amplitude_to_db(spectrogram, reference=100.0)
        self.assertTrue(np.all(np.isfinite(db)))

    def test_amplitude_to_db_invalid_reference(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), reference=-1.0)

    def test_amplitude_to_db_zero_reference(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), reference=0.0)

    def test_amplitude_to_db_invalid_min_value(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), min_value=0.0)

    def test_amplitude_to_db_negative_min_value(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), min_value=-0.1)

    def test_amplitude_to_db_invalid_db_range(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), db_range=0.0)

    def test_amplitude_to_db_negative_db_range(self):
        from paddleformers.transformers.audio_utils import amplitude_to_db

        with self.assertRaises(ValueError):
            amplitude_to_db(np.array([[1.0]]), db_range=-10.0)

    def test_spectrogram_basic(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=1.0)
        self.assertEqual(spec.ndim, 2)

    def test_spectrogram_basic_centered(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=1.0)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[0], 201)

    def test_spectrogram_power_two(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=2.0)
        self.assertEqual(spec.ndim, 2)
        self.assertFalse(np.iscomplexobj(spec))

    def test_spectrogram_with_mel_filters(self):
        from paddleformers.transformers.audio_utils import (
            mel_filter_bank,
            spectrogram,
            window_function,
        )

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        mel_filters = mel_filter_bank(
            num_frequency_bins=201,
            num_mel_filters=40,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
        )
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, mel_filters=mel_filters)
        self.assertEqual(spec.shape[0], 40)

    def test_spectrogram_with_log_mel(self):
        from paddleformers.transformers.audio_utils import (
            mel_filter_bank,
            spectrogram,
            window_function,
        )

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        mel_filters = mel_filter_bank(
            num_frequency_bins=201,
            num_mel_filters=40,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=16000,
        )
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, mel_filters=mel_filters, log_mel="log")
        self.assertEqual(spec.shape[0], 40)

    def test_spectrogram_log10(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=1.0, log_mel="log10")
        self.assertTrue(np.all(np.isfinite(spec)))

    def test_spectrogram_db_amplitude(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=1.0, log_mel="dB")
        self.assertTrue(np.all(np.isfinite(spec)))

    def test_spectrogram_db_power(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=2.0, log_mel="dB")
        self.assertTrue(np.all(np.isfinite(spec)))

    def test_spectrogram_invalid_power_for_db(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160, power=3.0, log_mel="dB")

    def test_spectrogram_invalid_log_mel_option(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160, power=1.0, log_mel="invalid")

    def test_spectrogram_frame_length_greater_than_fft(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160, fft_length=200)

    def test_spectrogram_window_length_mismatch(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(200, name="hann", periodic=True, frame_length=200, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160)

    def test_spectrogram_invalid_hop_length(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=0)

    def test_spectrogram_negative_hop_length(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=-1)

    def test_spectrogram_multidim_input(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(2, 1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160)

    def test_spectrogram_complex_input(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.complex64)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        with self.assertRaises(ValueError):
            spectrogram(waveform, window, frame_length=400, hop_length=160)

    def test_spectrogram_power_none(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, power=None)
        self.assertTrue(np.iscomplexobj(spec))

    def test_spectrogram_no_center(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, center=False, power=1.0)
        self.assertEqual(spec.ndim, 2)

    def test_spectrogram_with_preemphasis(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, preemphasis=0.97, power=1.0)
        self.assertEqual(spec.ndim, 2)

    def test_spectrogram_onesided_false(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, onesided=False, power=1.0)
        self.assertEqual(spec.shape[0], 400)

    def test_spectrogram_custom_fft_length(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(waveform, window, frame_length=400, hop_length=160, fft_length=512)
        self.assertEqual(spec.shape[0], 257)

    def test_spectrogram_custom_dtype(self):
        from paddleformers.transformers.audio_utils import spectrogram, window_function

        waveform = np.random.randn(1600).astype(np.float32)
        window = window_function(400, name="hann", periodic=True, frame_length=400, center=False)
        spec = spectrogram(
            waveform, window, frame_length=400, hop_length=160, power=1.0, log_mel="log", dtype=np.float64
        )
        self.assertEqual(spec.dtype, np.float64)

    def test_get_mel_filter_banks_deprecated(self):
        from paddleformers.transformers.audio_utils import get_mel_filter_banks

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_mel_filter_banks(
                nb_frequency_bins=257,
                nb_mel_filters=80,
                frequency_min=0.0,
                frequency_max=8000.0,
                sample_rate=16000,
            )
            self.assertTrue(len(w) > 0)
            self.assertTrue(any("deprecated" in str(x.message).lower() for x in w))
            self.assertEqual(result.shape, (257, 80))

    def test_fram_wave_deprecated(self):
        from paddleformers.transformers.audio_utils import fram_wave

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            waveform = np.random.randn(1600).astype(np.float32)
            frames = fram_wave(waveform, hop_length=160, fft_window_size=400, center=True)
            self.assertTrue(len(w) > 0)
            self.assertTrue(any("deprecated" in str(x.message).lower() for x in w))
            self.assertEqual(frames.ndim, 2)
            self.assertEqual(frames.shape[1], 400)

    def test_fram_wave_no_center(self):
        from paddleformers.transformers.audio_utils import fram_wave

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            waveform = np.random.randn(1600).astype(np.float32)
            frames = fram_wave(waveform, hop_length=160, fft_window_size=400, center=False)
            self.assertEqual(frames.ndim, 2)
            self.assertEqual(frames.shape[1], 400)

    def test_fram_wave_short_waveform(self):
        from paddleformers.transformers.audio_utils import fram_wave

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            waveform = np.random.randn(1600).astype(np.float32)
            frames = fram_wave(waveform, hop_length=160, fft_window_size=400, center=False)
            self.assertEqual(frames.ndim, 2)
            self.assertEqual(frames.shape[1], 400)

    def test_stft_deprecated(self):
        from paddleformers.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            frames = np.random.randn(10, 400).astype(np.float32)
            spec = stft(frames, np.hanning(400 + 1)[:-1])
            self.assertTrue(len(w) > 0)
            self.assertTrue(any("deprecated" in str(x.message).lower() for x in w))
            self.assertEqual(spec.ndim, 2)

    def test_stft_with_fft_window_size(self):
        from paddleformers.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            frames = np.random.randn(10, 400).astype(np.float32)
            spec = stft(frames, np.hanning(400 + 1)[:-1], fft_window_size=512)
            self.assertEqual(spec.ndim, 2)
            self.assertEqual(spec.shape[0], 257)

    def test_stft_fft_smaller_than_frame(self):
        from paddleformers.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            frames = np.random.randn(10, 400).astype(np.float32)
            with self.assertRaises(ValueError):
                stft(frames, np.hanning(400 + 1), fft_window_size=200)

    def test_stft_no_window(self):
        from paddleformers.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            frames = np.random.randn(10, 400).astype(np.float32)
            spec = stft(frames, None)
            self.assertEqual(spec.ndim, 2)

    def test_stft_none_fft_size(self):
        from paddleformers.transformers.audio_utils import stft

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            frames = np.random.randn(10, 400).astype(np.float32)
            spec = stft(frames, np.hanning(400 + 1)[:-1])
            self.assertEqual(spec.ndim, 2)
            self.assertEqual(spec.shape[0], 201)


if __name__ == "__main__":
    unittest.main()

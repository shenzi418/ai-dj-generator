import librosa
import numpy as np
import pytest

from pipeline import audio_utils as au


def click_track(bpm: float, seconds: float, sr: int) -> np.ndarray:
    times = np.arange(0.0, seconds, 60.0 / bpm)
    return librosa.clicks(times=times, sr=sr, length=int(seconds * sr)).astype(np.float32)


def test_clamp():
    assert au.clamp(5, 0, 1) == 1.0
    assert au.clamp(-5, 0, 1) == 0.0
    assert au.clamp(0.25, 0, 1) == 0.25


def test_ensure_mono_averages_channels():
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1)
    np.testing.assert_allclose(au.ensure_mono(stereo), np.full(4, 0.5))


def test_normalize_peak_only_attenuates():
    loud = np.array([0.0, 2.0, -4.0], dtype=np.float32)
    assert np.max(np.abs(au.normalize_peak(loud, peak=0.9))) == pytest.approx(0.9)
    quiet = np.array([0.1, -0.2], dtype=np.float32)
    np.testing.assert_allclose(au.normalize_peak(quiet), quiet)


def test_apply_edge_fades_zeroes_edges():
    y = au.apply_edge_fades(np.ones(1000, dtype=np.float32), sr=1000, fade_ms=100)
    assert y[0] == 0.0 and y[-1] == 0.0
    assert y[500] == 1.0


def test_ensure_length_pads_and_truncates():
    assert au.ensure_length(np.ones(3), 5).tolist() == [1, 1, 1, 0, 0]
    assert au.ensure_length(np.ones(5), 2).size == 2


def test_crossfade_equal_length_endpoints():
    out = au.crossfade_equal_length(np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32))
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(0.0)


def test_choose_beats():
    beats = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    assert au.choose_nearest_beat(beats, 1.2) == pytest.approx(1.0)
    assert au.choose_first_beat_after(beats, 1.2) == pytest.approx(1.5)
    assert au.choose_first_beat_after(np.array([]), 3.0) == 3.0


@pytest.mark.parametrize("bpm", [100.0, 128.0])
def test_estimate_bpm_on_click_track(bpm):
    sr = 22050
    tempo, beats = au.estimate_bpm_and_beats(click_track(bpm, 20.0, sr), sr)
    assert tempo is not None
    assert tempo == pytest.approx(bpm, rel=0.05)
    assert beats.size > 10


def test_write_and_decode_roundtrip(tmp_path):
    sr = 16000
    y = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)).astype(np.float32)
    path = str(tmp_path / "sub" / "tone.wav")
    au.write_wav(path, y, sr)
    seg, read_sr = au.decode_segment(path, start_sec=0.5, duration_sec=1.0, sr=sr)
    assert read_sr == sr
    assert seg.size == pytest.approx(sr, abs=32)

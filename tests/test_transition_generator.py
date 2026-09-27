import numpy as np
import pytest
import soundfile as sf

from pipeline import transition_generator as tg
from tests.test_audio_utils import click_track


def test_beats_seconds_roundtrip():
    assert tg._beats_to_seconds(32, 120.0) == pytest.approx(16.0)
    assert tg._seconds_to_beats(16.0, 120.0) == pytest.approx(32.0)


@pytest.mark.parametrize(
    "ref, candidate, expected",
    [(128.0, 64.0, 128.0), (70.0, 140.0, 70.0), (124.0, 126.0, 126.0)],
)
def test_resolve_half_double_tempo(ref, candidate, expected):
    assert tg._resolve_half_double_tempo(ref, candidate) == pytest.approx(expected)


def test_phrase_lock_snaps_to_whole_beats():
    shape = tg._phrase_lock_transition_shape(pre_sec=6.1, seam_sec=15.3, post_sec=5.7, bpm=120.0)
    for key in ("pre", "post"):
        assert shape["debug"][key]["locked_beats"] % 4 == 0
    assert shape["debug"]["seam"]["locked_beats"] % 8 == 0
    assert shape["seam_sec"] == pytest.approx(16.0)


def test_build_caption_uses_preset_and_instruction():
    assert tg._build_caption("Ambient Wash", "") == tg.PLUGIN_PRESETS["Ambient Wash"]
    caption = tg._build_caption("Unknown", "more hi-hats")
    assert caption.startswith(tg.PLUGIN_PRESETS["Smooth Blend"])
    assert caption.endswith("more hi-hats")


def test_output_stem_is_deterministic(tmp_path):
    kwargs = dict(song_a_path="/x/Song A.mp3", song_b_path="/y/song-b.wav", seed=7, output_dir=str(tmp_path))
    stem1 = tg._deterministic_stem(tg.TransitionRequest(**kwargs))
    stem2 = tg._deterministic_stem(tg.TransitionRequest(**kwargs))
    stem3 = tg._deterministic_stem(tg.TransitionRequest(**{**kwargs, "seed": 8}))
    assert stem1 == stem2
    assert stem1 != stem3
    assert stem1.startswith("transition_Song_A_to_song-b_")


def test_prepare_rough_transition_end_to_end(tmp_path):
    """Runs the full analysis stage (BPM, cue selection, tempo matching, seam build) without ACE-Step."""
    sr = tg.DEFAULT_TARGET_SR
    song_a = tmp_path / "a.wav"
    song_b = tmp_path / "b.wav"
    sf.write(song_a, click_track(100.0, 40.0, sr), sr)
    sf.write(song_b, click_track(128.0, 40.0, sr), sr)

    request = tg.TransitionRequest(
        song_a_path=str(song_a),
        song_b_path=str(song_b),
        analysis_sec=30.0,
        transition_bars=4,
        output_dir=str(tmp_path / "out"),
    )
    rough = tg._prepare_rough_transition(request)

    assert rough["bpm_a"] == pytest.approx(100.0, rel=0.03)
    assert rough["stretch_rate"] == pytest.approx(rough["bpm_a"] / rough["bpm_b_for_alignment"])
    assert rough["seam_n"] == int(round(rough["seam_sec"] * sr))
    assert rough["rough_seam"].size == rough["seam_n"]
    assert np.isfinite(rough["rough_stitched"]).all()

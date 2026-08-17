"""
Tests for the real Windows microphone/speaker hardware adapter
``SoundDeviceHardware`` (aura/interaction/voice/hardware.py).

These run FULLY OFFLINE. ``sounddevice`` and ``numpy`` (the only two libraries
the adapter uses) are injected as tiny fakes via the ``sd_module`` /
``np_module`` dependency-injection hooks -- no microphone, speaker, audio
driver, or OS-specific code is ever touched. Missing-engine error paths are
simulated by setting the module to ``None`` in ``sys.modules``, exactly like the
voice provider tests do, so they stay deterministic whether or not the real
libraries happen to be installed.
"""

import io
import struct
import sys
import types
import wave

import numpy as np

import pytest

from aura.interaction import (
    Audio,
    SoundDeviceHardware,
    SpeechError,
)
from aura.interaction.base import InteractionEnd


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePCM:
    """Stands in for a numpy int16 array returned by ``stream.read()``."""

    def __init__(self, data: bytes) -> None:
        self._data = bytes(data)

    def tobytes(self) -> bytes:
        return self._data


class FakeStream:
    """A fake ``sounddevice.InputStream`` that never touches real hardware."""

    def __init__(
        self,
        chunks,
        fail_read: bool = False,
        interrupt: bool = False,
    ) -> None:
        self._chunks = list(chunks)
        self._fail_read = fail_read
        self._interrupt = interrupt
        self.started = False
        self.closed = False
        self.read_count = 0

    def start(self) -> None:
        self.started = True

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc) -> bool:
        self.closed = True
        return False  # never suppress exceptions

    def read(self, size: int):
        self.read_count += 1
        if self._interrupt:
            raise KeyboardInterrupt
        if self._fail_read:
            raise RuntimeError("audio device disconnected")
        if self._chunks:
            return self._chunks.pop(0), 0
        # No more queued audio: hand back an empty buffer, which the adapter
        # treats as silence / produces no frames.
        return FakePCM(b""), 0


def make_sd(stream_factory=None, interrupt_play: bool = False, fail_play: bool = False):
    """Build a fake ``sounddevice`` module with ``InputStream`` and ``play``."""
    sd = types.SimpleNamespace()
    sd.InputStream = stream_factory or (lambda **kw: FakeStream([]))
    sd.play_calls: list[tuple] = []

    def _play(arr, samplerate, blocking: bool = True) -> None:
        if interrupt_play:
            raise KeyboardInterrupt
        if fail_play:
            raise RuntimeError("no output device")
        sd.play_calls.append((arr, samplerate, blocking))

    sd.play = _play
    return sd


def make_np():
    """Build a fake ``numpy`` module exposing just what playback needs."""
    np = types.SimpleNamespace()
    np.uint8 = "uint8"
    np.int16 = "int16"
    np.int32 = "int32"
    np.frombuffer_calls: list[tuple] = []

    class _Arr:
        def __init__(self, data: bytes) -> None:
            self._data = bytes(data)

        def reshape(self, *_shape) -> "_Arr":
            return self

        def __len__(self) -> int:
            return len(self._data)

    def _frombuffer(buffer, dtype):
        data = bytes(buffer)
        np.frombuffer_calls.append((data, dtype))
        return _Arr(data)

    np.frombuffer = _frombuffer
    return np



def pcm_chunk(amplitude: int = 1000, size: int = 32) -> FakePCM:
    """A non-silent 16-bit little-endian PCM chunk (amplitude 1000 >> 0.005)."""
    return FakePCM(struct.pack(f"<{size}h", *([amplitude] * size)))


def silence_chunk(size: int = 32) -> FakePCM:
    return FakePCM(b"\x00\x00" * size)


def make_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Encode raw 16-bit PCM as a minimal RIFF/WAVE container."""
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * 2,
        channels * 2,
        16,
    )
    data_chunk = struct.pack("<4sI", b"data", len(pcm)) + pcm
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data_chunk))
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk


def decode_wav(data: bytes) -> tuple[bytes, int, int]:
    """Decode a WAV container back to ``(pcm, sample_rate, channels)``."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return (
            w.readframes(w.getnframes()),
            w.getframerate(),
            w.getnchannels(),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_configurable_sample_rate_channels_and_recording_limit() -> None:
    hw = SoundDeviceHardware(
        sample_rate=48000,
        channels=2,
        recording_limit=3.0,
    )
    assert hw.sample_rate == 48000
    assert hw.channels == 2
    assert hw.recording_limit == 3.0


def test_defaults_target_vosk() -> None:
    hw = SoundDeviceHardware()
    assert hw.sample_rate == 16000  # matches Vosk models
    assert hw.channels == 1
    assert hw.recording_limit == 5.0


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------


CHUNK_SAMPLES = 32  # frames per FakeStream chunk (matches pcm_chunk/silence_chunk)


def test_capture_records_until_trailing_silence_after_speech() -> None:
    speech = [pcm_chunk(amplitude=2000) for _ in range(3)]
    # Default silence_duration = 0.6s: at 16k/1024 chunk size that is
    # int(0.6 * 16000 / 1024) = 9 consecutive trailing-silence chunks required
    # to end the utterance AFTER speech has started.
    trailing_silence = [silence_chunk() for _ in range(9)]
    stream = FakeStream(speech + trailing_silence)
    sd = make_sd()
    sd.InputStream = lambda **kw: stream
    hw = SoundDeviceHardware(sd_module=sd)

    result = hw.capture()

    assert isinstance(result, Audio)
    assert result.format == "wav"
    pcm, rate, channels = decode_wav(result.data)
    assert rate == 16000
    assert channels == 1
    # 3 speech + 9 consecutive trailing-silence chunks were read before the break.
    assert stream.read_count == 12
    assert len(pcm) == 12 * CHUNK_SAMPLES * 2
    assert stream.started
    assert stream.closed
    assert stream.read_count < 78  # stopped on trailing silence, not the limit


def test_capture_ignores_leading_silence_before_speech() -> None:
    # Regression: leading ambient silence (the quiet moment before the user
    # begins to speak) must NEVER end the capture. Feed far more leading silence
    # than silence_chunks_needed (20 >> 9): a buggy detector that ended on ANY
    # silence would stop at read #9 having captured only silence.
    leading = [silence_chunk() for _ in range(20)]
    speech = [pcm_chunk(amplitude=2000) for _ in range(3)]
    trailing = [silence_chunk() for _ in range(9)]
    stream = FakeStream(leading + speech + trailing)
    sd = make_sd()
    sd.InputStream = lambda **kw: stream
    hw = SoundDeviceHardware(sd_module=sd)

    result = hw.capture()

    # Capture survived the leading silence, caught the speech, and ended only on
    # the trailing silence after it: 20 leading + 3 speech + 9 trailing reads.
    assert stream.read_count == 32
    pcm, *_ = decode_wav(result.data)
    assert len(pcm) == 32 * CHUNK_SAMPLES * 2





def test_capture_is_bounded_by_recording_limit() -> None:
    # Feed continuous speech well beyond the hard limit.
    stream = FakeStream([pcm_chunk(amplitude=2000) for _ in range(200)])
    hw = SoundDeviceHardware(sd_module=make_sd(stream_factory=lambda **kw: stream))
    assert hw.recording_limit * hw.sample_rate // 1024 == 78

    result = hw.capture()

    assert stream.read_count == 78  # never exceeded the bounded max
    assert len(decode_wav(result.data)[0]) == 78 * CHUNK_SAMPLES * 2


def test_capture_passes_known_sample_rate_and_channels_to_stream() -> None:
    recorded: dict = {}

    def factory(**kw) -> "FakeStream":
        recorded.update(kw)
        return FakeStream([pcm_chunk(), silence_chunk()])

    hw = SoundDeviceHardware(
        sample_rate=22050,
        channels=1,
        sd_module=make_sd(stream_factory=factory),
    )
    hw.capture()

    assert recorded["samplerate"] == 22050
    assert recorded["channels"] == 1
    assert recorded["dtype"] == "int16"


def test_capture_raises_speech_error_when_mic_cannot_open() -> None:
    def factory(**_kw) -> "FakeStream":
        raise RuntimeError("PortAudio error: no input device")

    hw = SoundDeviceHardware(sd_module=make_sd(stream_factory=factory))
    with pytest.raises(SpeechError):
        hw.capture()


def test_capture_raises_speech_error_on_read_failure() -> None:
    stream = FakeStream([], fail_read=True)
    hw = SoundDeviceHardware(sd_module=make_sd(stream_factory=lambda **kw: stream))
    with pytest.raises(SpeechError):
        hw.capture()


def test_capture_propagates_interaction_end_on_keyboard_interrupt() -> None:
    stream = FakeStream([], interrupt=True)
    hw = SoundDeviceHardware(sd_module=make_sd(stream_factory=lambda **kw: stream))
    with pytest.raises(InteractionEnd):
        hw.capture()


def test_capture_raises_speech_error_when_no_audio_produced() -> None:
    # Only empty buffers -> no frames -> no usable audio.
    hw = SoundDeviceHardware(sd_module=make_sd())
    with pytest.raises(SpeechError):
        hw.capture()


def test_capture_raises_speech_error_when_sounddevice_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    hw = SoundDeviceHardware()  # no injected module -> lazy import fails
    with pytest.raises(SpeechError):
        hw.capture()




# ---------------------------------------------------------------------------
# playback()
# ---------------------------------------------------------------------------


def test_playback_plays_wav_through_default_device() -> None:
    pcm = struct.pack("<10h", *([250] * 10))
    wav = make_wav(pcm, sample_rate=16000, channels=1)
    sd = make_sd()
    np = make_np()
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format="wav"))

    assert len(sd.play_calls) == 1
    arr, samplerate, blocking = sd.play_calls[0]
    assert arr._data == pcm  # RAW PCM samples handed to sounddevice
    assert samplerate == 16000
    assert blocking is True


def test_playback_decodes_int16_dtype_for_16_bit_pcm() -> None:
    hw = SoundDeviceHardware(sd_module=make_sd(), np_module=make_np())
    hw.playback(Audio(data=make_wav(b"\x00\x01\x00\x02"), format="wav"))
    # np.frombuffer was asked for int16 (the most common TTS output width).
    assert hw._np_module.frombuffer_calls[0][1] == "int16"


def test_playback_reshapes_stereo_channels() -> None:
    pcm = struct.pack("<4h", 100, 100, 200, 200)  # 2 frames x 2 channels
    hw = SoundDeviceHardware(sd_module=make_sd(), np_module=make_np())
    hw.playback(Audio(data=make_wav(pcm, channels=2), format="wav"))
    assert hw._np_module.frombuffer_calls[0][1] == "int16"


def test_playback_raises_speech_error_on_failed_wav_decode() -> None:
    hw = SoundDeviceHardware(sd_module=make_sd(), np_module=make_np())
    with pytest.raises(SpeechError):
        hw.playback(Audio(data=b"RIFF-not-a-wav", format="wav"))
    assert hw._sd_module.play_calls == []  # nothing ever reached the speaker


def test_playback_raises_speech_error_on_empty_wav() -> None:
    hw = SoundDeviceHardware(sd_module=make_sd(), np_module=make_np())
    # A WAV header with zero frames holds no audio.
    with pytest.raises(SpeechError):
        hw.playback(Audio(data=make_wav(b""), format="wav"))


def test_playback_raises_interaction_end_on_keyboard_interrupt() -> None:
    hw = SoundDeviceHardware(
        sd_module=make_sd(interrupt_play=True),
        np_module=make_np(),
    )
    with pytest.raises(InteractionEnd):
        hw.playback(Audio(data=make_wav(b"\x00\x01"), format="wav"))


def test_playback_raises_speech_error_on_playback_failure() -> None:
    hw = SoundDeviceHardware(
        sd_module=make_sd(fail_play=True),
        np_module=make_np(),
    )
    with pytest.raises(SpeechError):
        hw.playback(Audio(data=make_wav(b"\x00\x01"), format="wav"))


def test_playback_raises_speech_error_when_numpy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    # sounddevice injected (never touches hardware), numpy left lazy -> fails.
    hw = SoundDeviceHardware(sd_module=make_sd(), np_module=None)
    with pytest.raises(SpeechError):
        hw.playback(Audio(data=make_wav(b"\x00\x01"), format="wav"))


# ---------------------------------------------------------------------------
# barge-in (interruptible playback)
# ---------------------------------------------------------------------------


def _barge_sd(speech_chunks, stream_active: bool = True, input_factory=None):
    """A fake ``sounddevice`` whose ``play`` is non-blocking and stoppable."""
    sd = types.SimpleNamespace()
    sd.InputStream = input_factory or (lambda **kw: FakeStream(speech_chunks))
    sd.play_calls: list[tuple] = []
    sd.stop_calls = 0
    sd.wait_calls = 0
    sd.active = stream_active

    def _play(arr, samplerate, blocking: bool = True) -> None:
        sd.play_calls.append((arr, samplerate, blocking))

    def _stop() -> None:
        sd.stop_calls += 1

    def _wait() -> None:
        sd.wait_calls += 1

    def _get_stream():
        return types.SimpleNamespace(active=sd.active)

    sd.play = _play
    sd.stop = _stop
    sd.wait = _wait
    sd.get_stream = _get_stream
    return sd


def _any_arr() -> object:
    """A dummy numpy array standing in for the decoded playback samples."""
    return make_np().frombuffer(b"\x00\x01\x00\x02", "int16")


def test_barge_in_config_defaults_off() -> None:
    assert SoundDeviceHardware().barge_in is False
    assert SoundDeviceHardware(barge_in=True).barge_in is True


def test_playback_preserves_blocking_mode_when_barge_in_off() -> None:
    # With barge-in disabled the verified baseline path is unchanged: playback
    # is a single blocking sd.play (existing behavior must not regress).
    sd = _barge_sd([])
    hw = SoundDeviceHardware(sd_module=sd, np_module=make_np(), barge_in=False)
    hw.playback(Audio(data=make_wav(b"\x00\x01\x00\x02"), format="wav"))
    assert sd.play_calls[0][2] is True  # blocking=True
    assert sd.stop_calls == 0  # no barge machinery touched


def test_playback_barge_in_stops_on_user_speech() -> None:
    # 3 voiced chunks == 0.2s sustain @ 16k/1024 -> barge triggers on the 3rd.
    speech = [pcm_chunk(amplitude=2000) for _ in range(3)]
    sd = _barge_sd(speech, stream_active=True)
    hw = SoundDeviceHardware(sd_module=sd, np_module=make_np(), barge_in=True)

    interrupted = hw._playback_bargeable(sd, _any_arr(), 16000)

    assert interrupted is True  # user speech interrupted the reply
    assert sd.play_calls and sd.play_calls[0][2] is False  # non-blocking play
    assert sd.stop_calls >= 1  # playback aborted/cleaned up


def test_playback_barge_in_does_not_trigger_when_user_is_silent() -> None:
    sd = _barge_sd([silence_chunk()], stream_active=False)
    hw = SoundDeviceHardware(sd_module=sd, np_module=make_np(), barge_in=True)

    interrupted = hw._playback_bargeable(sd, _any_arr(), 16000)

    # Playback ran to its natural end (stream became inactive) -- no barge.
    assert interrupted is False
    assert sd.stop_calls >= 1  # teardown still happens


def test_playback_barge_in_falls_back_when_mic_unavailable() -> None:
    def no_input(**kw) -> None:
        raise RuntimeError("no input device")

    sd = _barge_sd([], input_factory=no_input)
    hw = SoundDeviceHardware(sd_module=sd, np_module=make_np(), barge_in=True)

    interrupted = hw._playback_bargeable(sd, _any_arr(), 16000)

    # Can't listen while speaking -> degrades to plain (drained) playback.
    assert interrupted is False
    assert sd.wait_calls == 1
    assert sd.stop_calls >= 1


# ---------------------------------------------------------------------------
# playback sample-rate resampling (the "ki-ka-pa" fix)
# ---------------------------------------------------------------------------


def test_playback_resamples_to_explicit_device_rate() -> None:
    # TTS @ 1000 Hz sent to a device forced at 2000 Hz must be resampled
    # (8 -> 16 int16 samples) and played at the device rate, not the WAV rate.
    pcm = np.arange(8, dtype=np.int16).tobytes()
    wav = make_wav(pcm, sample_rate=1000)
    sd = _barge_sd([])
    hw = SoundDeviceHardware(sd_module=sd, np_module=np, output_sample_rate=2000)

    hw.playback(Audio(data=wav, format="wav"))

    arr, rate, blocking = sd.play_calls[0]
    assert rate == 2000  # played at the device rate, not 1000
    assert blocking is True
    assert arr.dtype == np.int16
    assert len(arr) == 16  # 8 samples * 2000/1000


def test_playback_auto_detects_device_sample_rate() -> None:
    # With no explicit rate, the device's native default_samplerate is used.
    sd = _barge_sd([])
    sd.default = types.SimpleNamespace(device=np.array([0, 1]))  # [input, output]
    sd.query_devices = lambda device=None: {"default_samplerate": 44100.0}
    pcm = np.arange(8, dtype=np.int16).tobytes()
    wav = make_wav(pcm, sample_rate=1000)
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format="wav"))

    arr, rate, _blocking = sd.play_calls[0]
    assert rate == 44100  # matched to the output device's native rate


def test_playback_skips_resample_when_rates_match() -> None:
    pcm = np.arange(8, dtype=np.int16).tobytes()
    wav = make_wav(pcm, sample_rate=44100)
    sd = _barge_sd([])
    sd.default = types.SimpleNamespace(device=np.array([0, 1]))
    sd.query_devices = lambda device=None: {"default_samplerate": 44100.0}
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format="wav"))

    arr, rate, _blocking = sd.play_calls[0]
    assert rate == 44100
    assert len(arr) == 8  # already matched -> untouched


def test_resample_audio_passthrough_when_rates_equal() -> None:
    arr = np.arange(10, dtype=np.int16)
    out = SoundDeviceHardware._resample_audio(np, arr, 22050, 22050)
    assert out is arr
    assert out.dtype == np.int16

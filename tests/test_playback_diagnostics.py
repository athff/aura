import types
import struct
import io
import wave
import numpy as np

import pytest

from aura.interaction import Audio, SoundDeviceHardware, SpeechError


# Helper to build minimal WAV

def make_wav(pcm: bytes, sample_rate: int = 22050, channels: int = 1) -> bytes:
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


class SimpleSD(types.SimpleNamespace):
    def __init__(self, *, default_samplerate: int | None = None, fail_play: bool = False):
        super().__init__()
        self.play_calls = []
        self.fail_play = fail_play
        if default_samplerate is not None:
            self.default = types.SimpleNamespace(device=[0, 1])
            self.query_devices = lambda device: {"default_samplerate": default_samplerate}
        else:
            self.default = types.SimpleNamespace(device=1)
            self.query_devices = lambda device: {}

    def InputStream(self, **kw):
        # not used in these tests
        class _Ctx:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self, size):
                return b"", 0
        return _Ctx()

    def play(self, arr, samplerate, blocking=True):
        if self.fail_play:
            raise RuntimeError("no output device")
        self.play_calls.append((arr, samplerate, blocking))


def test_resampling_produces_nonzero_pcm_when_required():
    # Create a short 22050Hz mono int16 PCM sequence with non-zero values
    pcm = struct.pack("<10h", *([1000] * 10))
    wav = make_wav(pcm, sample_rate=22050, channels=1)
    sd = SimpleSD(default_samplerate=44100)
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format="wav"))

    # Ensure we attempted playback and samplerate used was 44100 (resampled)
    assert len(sd.play_calls) >= 1
    arr, sr, _ = sd.play_calls[0]
    assert int(sr) == 44100
    arr_np = np.asarray(arr)
    assert arr_np.size > 0
    assert np.any(arr_np != 0)


def test_playback_exceptions_are_reported():
    pcm = struct.pack("<10h", *([1000] * 10))
    wav = make_wav(pcm, sample_rate=16000, channels=1)
    sd = SimpleSD(default_samplerate=16000, fail_play=True)
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    with pytest.raises(SpeechError):
        hw.playback(Audio(data=wav, format="wav"))

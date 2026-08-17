import types
import io
import wave
import struct
import math

import numpy as np

from aura.interaction import Audio, SoundDeviceHardware


class FakeSD(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.play_calls = []
        # Provide an OutputStream too to simulate both being present
        def _make_stream(*args, **kwargs):
            class S:
                def __init__(self):
                    self.writes = []
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
                def write(self, data):
                    import numpy as _np
                    a = _np.asarray(data)
                    self.writes.append(a.shape[0])
            return S()
        self.OutputStream = _make_stream

    def play(self, arr, samplerate, blocking=True):
        self.play_calls.append((arr, samplerate, blocking))

    def wait(self):
        pass


def make_sine_wav(freq=440, duration=2.0, sample_rate=22050, amplitude=8000, channels=1):
    n = int(sample_rate * duration)
    pcm = bytearray()
    for i in range(n):
        v = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
        pcm += struct.pack('<h', v)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    return buf.getvalue()


def test_prefers_sd_play_when_available():
    wav = make_sine_wav(duration=2.0)
    sd = FakeSD()
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format='wav'))

    # sd.play should have been called once
    assert len(sd.play_calls) == 1
    arr, sr, blocking = sd.play_calls[0]
    assert blocking is True
    assert int(sr) in (22050, 44100)

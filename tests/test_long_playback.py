import struct
import types
import io
import wave
import math

import numpy as np

from aura.interaction import Audio, SoundDeviceHardware


# Build a fake sounddevice with wait() tracking
class FakeSD(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.play_calls = []
        self.wait_called = False
        self.stop_called = False

    def play(self, arr, samplerate, blocking=True):
        self.play_calls.append((arr, samplerate, blocking))

    def wait(self):
        self.wait_called = True

    def stop(self):
        self.stop_called = True


def make_long_sine_wav(freq=440, duration=20.0, sample_rate=22050, amplitude=8000, channels=1):
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


def test_long_playback_blocks_and_waits():
    # 20s sine at 22050 Hz -> 441000 samples
    wav = make_long_sine_wav(duration=20.0)
    sd = FakeSD()
    hw = SoundDeviceHardware(sd_module=sd, np_module=np)

    hw.playback(Audio(data=wav, format='wav'))

    # Ensure play was called once
    assert len(sd.play_calls) == 1
    arr, sr, blocking = sd.play_calls[0]
    assert int(sr) in (22050, 44100)
    # Ensure wait() was called to block until playback completion
    assert sd.wait_called is True
    # Ensure stop() was not called by normal playback
    assert sd.stop_called is False
    # Ensure array contains samples
    arr_np = np.asarray(arr)
    assert arr_np.size > 0
    # Ensure duration approximates 20 seconds
    frames = int(arr_np.shape[0]) if hasattr(arr_np, 'shape') else len(arr_np)
    # allow one-sample tolerance
    assert abs(frames - int(22050 * 20.0)) <= 2

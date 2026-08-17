"""
interaction/voice/hardware.py
-----------------------------
Real Windows microphone capture + speaker playback using ``sounddevice``.

This module is the ONLY place in AURA that imports ``sounddevice`` / ``numpy``.
It is deliberately isolated from ``VoiceInteraction``, ``VoskSpeechInput``,
``KokoroSpeechOutput``, ``Session``, and ``AuraEngine``.  It provides two
callables -- ``capture`` (one utterance from the microphone) and ``playback``
(one ``Audio`` payload to the speakers) -- that match the
``AudioCapture`` / ``AudioPlayback`` types already used by
``VoiceInteraction``:

    hardware = SoundDeviceHardware()
    voice = VoiceInteraction(
        speech_input=...,
        speech_output=...,
        capture=hardware.capture,      # AudioCapture
        playback=hardware.playback,    # AudioPlayback
    )

``sounddevice`` and ``numpy`` are imported **lazily** (only when
``capture``/``playback`` is actually called).  Importing this module never
requires audio drivers or OS-specific libraries.  When the library or hardware
is unavailable, ``SpeechError`` is raised so callers can fall back gracefully.

No wake-word detection, continuous listening, or streaming is involved:
``capture`` records exactly one bounded utterance per call, and ``playback``
emits exactly one ``Audio`` payload per call.
"""

import io
import struct
import wave
from time import perf_counter
from typing import Any

from aura.core.logging import get_logger
from aura.interaction.base import InteractionEnd
from aura.interaction.voice.base import Audio, SpeechError

# 16-bit PCM = 2 bytes per sample (the format Vosk models and synthesized output use).
_SAMPLE_WIDTH = 2

logger = get_logger(__name__)


def _encode_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Encode raw 16-bit little-endian PCM bytes into a RIFF/WAVE container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(_SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _decode_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """Decode a WAV container.

    Returns ``(pcm_bytes, sample_rate, channels, sample_width)``.
    Raises ``SpeechError`` if *data* is not a valid WAV payload.
    """
    buf = io.BytesIO(data)
    try:
        with wave.open(buf, "rb") as w:
            sample_rate = w.getframerate()
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            pcm = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise SpeechError(
            f"Cannot decode WAV payload for playback: {exc}"
        ) from exc

    if not pcm:
        raise SpeechError(
            "Cannot decode empty or invalid WAV payload for playback"
        )
    return pcm, sample_rate, channels, sample_width


def _rms16(pcm: bytes) -> float:
    """Normalised RMS amplitude of 16-bit signed PCM bytes (pure Python).

    Returns a float in [0.0, 1.0] where 1.0 is maximum possible amplitude
    (32 768 for signed int16).  No numpy required.
    """
    if not pcm:
        return 0.0
    values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    sum_sq = sum(v * v for v in values)
    return (sum_sq / len(values)) ** 0.5 / 32768.0


# Small numpy helpers used by playback diagnostics/fallbacks. Kept local to
# hardware so tests can inject fake np modules if needed.
def _np_rms(arr: Any) -> float:
    """Return RMS (not normalised) for a numpy-like array. Best-effort."""
    try:
        a = arr.astype("float64")
        # flatten all channels
        a_flat = a.reshape(-1)
        return float((a_flat * a_flat).mean() ** 0.5)
    except Exception:
        # Fallback: try manual iteration if array-like
        try:
            vals = list(arr)
            if not vals:
                return 0.0
            total = 0.0
            count = 0
            for v in vals:
                total += (float(v) * float(v))
                count += 1
            return (total / count) ** 0.5
        except Exception:
            return 0.0


def _ensure_float32_for_playback(np_mod: Any, arr: Any) -> Any:
    """Convert integer PCM array to float32 in [-1.0, 1.0] for sounddevice.

    Works with int16, int32, uint8. Returns original array if already float32.
    """
    try:
        arr_np = np_mod.asarray(arr)
    except Exception:
        # As a last resort, return arr unchanged so caller can surface an error
        return arr

    try:
        dtype = arr_np.dtype
    except Exception:
        # Unknown dtype; attempt to coerce to float32
        try:
            return np_mod.asarray(arr_np, dtype="float32")
        except Exception:
            return arr

    kind = getattr(dtype, "kind", "i")
    if kind == "f" and getattr(dtype, "itemsize", 4) == 4:
        return arr_np.astype("float32")

    # Integer cases
    if kind in ("i", "u"):
        bits = dtype.itemsize * 8
        # Signed int: scale by max positive value
        if kind == "i":
            maxval = float(2 ** (bits - 1))
            out = arr_np.astype("float32") / maxval
            # clip to [-1, 1]
            out = np_mod.clip(out, -1.0, 1.0)
            return out
        else:
            # unsigned (e.g., uint8): convert from 0..max -> -1..1
            maxval = float(2 ** bits - 1)
            out = (arr_np.astype("float32") / maxval) * 2.0 - 1.0
            out = np_mod.clip(out, -1.0, 1.0)
            return out

    # Unknown kind: try to cast to float32
    try:
        return arr_np.astype("float32")
    except Exception:
        return arr


# ---------------------------------------------------------------------------
# Hardware adapter
# ---------------------------------------------------------------------------


class SoundDeviceHardware:
    """Real Windows microphone / speaker adapter backed by ``sounddevice``.

    Parameters
    ----------
    sample_rate:
        Capture / playback sample rate in Hz.  Defaults to 16 000 which
        matches what Vosk models expect.
    channels:
        Number of audio channels.  ``1`` = mono (default).
    recording_limit:
        Hard upper bound on capture duration, in seconds.  Recording always
        stops after this many seconds even if speech continues.
    silence_threshold:
        Normalised RMS below which a chunk is considered "silence"
        (0.0 - 1.0).  Default ``0.005`` approximates a very quiet room.
    silence_duration:
        How many *consecutive* seconds of trailing silence (after speech has
        started) must occur before the utterance is considered finished.
        Default ``0.6`` s -- long enough to capture complete natural sentences
        and short mid-sentence pauses, yet still a responsive end-of-speech
        trigger. Leading (pre-speech) silence never counts.
    chunk_size:
        Frames read per ``stream.read()`` call.  Smaller = more responsive
        silence detection, larger = lower CPU.  Default 1 024.
    barge_in:
        When True, playback listens on the microphone while speaking and
        stops (truncates) the current reply as soon as sustained user speech is
        detected, so the user can interrupt AURA ("barge-in"). Single-threaded
        and non-overlapping: after a barge the caller returns and the normal
        ``capture`` is used for the new utterance. Default ``False`` so the
        verified baseline is unchanged (enabling it makes the mic hear AURA's
        own speaker echo -- see ``barge_sustain`` and the module docs).
    barge_sustain:
        Seconds of *consecutive* voiced audio above ``silence_threshold``
        required before a barge-in interrupt fires. Tuned to ignore brief
        clicks/breath. Default ``0.2``.
    output_sample_rate:
        Explicit sample rate to play back at. If ``None`` (default), the
        playback sample rate is taken from the WAV itself and, when it differs
        from the output device's native ``default_samplerate`` (e.g. Kokoro's
        24 000 Hz vs a 44 100 Hz device), the audio is resampled to the device
        rate so it is NOT played back at the wrong speed (a real bug on MME
        hosts). Leave ``None`` to auto-detect.
    sd_module:
        Optional pre-built ``sounddevice`` module for dependency injection
        (e.g. a fake in tests).  When ``None`` the module is lazily imported
        on first use.
    np_module:
        Optional pre-built ``numpy`` module for dependency injection.
        Only needed for ``playback``; lazily imported when ``None``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        recording_limit: float = 5.0,
        silence_threshold: float = 0.005,
        silence_duration: float = 0.6,
        chunk_size: int = 1024,
        barge_in: bool = False,
        barge_sustain: float = 0.2,
        output_sample_rate: int | None = None,
        sd_module: Any | None = None,
        np_module: Any | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording_limit = recording_limit
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._chunk_size = chunk_size
        self._barge_in = barge_in
        self._barge_sustain = barge_sustain
        self._output_sample_rate = output_sample_rate
        self._sd_module = sd_module
        self._np_module = np_module

    @property
    def sample_rate(self) -> int:
        """Configured capture / playback sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def recording_limit(self) -> float:
        """Hard upper bound on capture duration, in seconds."""
        return self._recording_limit

    @property
    def barge_in(self) -> bool:
        """Whether playback listens for user speech and is interruptible."""
        return self._barge_in

    @property
    def output_sample_rate(self) -> int | None:
        """Explicit playback sample rate override, or ``None`` for auto."""
        return self._output_sample_rate

    def _get_sd(self) -> Any:
        """Return the injected ``sounddevice`` module or import it lazily."""
        if self._sd_module is not None:
            return self._sd_module
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SpeechError(
                "sounddevice is not installed: run "
                "'pip install -r requirements-voice.txt' to enable "
                "microphone and speaker hardware"
            ) from exc
        self._sd_module = sd
        return sd

    def _get_np(self) -> Any:
        """Return the injected ``numpy`` module or import it lazily."""
        if self._np_module is not None:
            return self._np_module
        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SpeechError(
                "numpy is not installed: run "
                "'pip install -r requirements-voice.txt' to enable "
                "speaker playback"
            ) from exc
        self._np_module = np
        return np


    # -- public API ----------------------------------------------------------

    def capture(self) -> Audio:
        """Record one bounded utterance from the microphone.

        Voice-activity-gated end-of-speech capture: the stream is read from the
        moment the mic opens, but the utterance is only *ended* by detecting
        ``silence_duration`` seconds of *trailing* silence AFTER at least one
        voiced (non-silent) chunk has been heard. Leading ambient silence before
        the user speaks NEVER ends the capture (that would finalize ~0.3 s of
        noise and yield an empty transcript). The capture stops when **either**:

        * ``silence_duration`` of trailing silence occurs after real speech, or
        * ``recording_limit`` seconds have elapsed (hard bound, which also
          bounds a silent wait before the user starts speaking).

        Returns
        -------
        Audio
            A WAV-encoded payload (``format="wav"``) ready for
            ``VoskSpeechInput.transcribe`` or ``StubSpeechInput``.

        Raises
        ------
        SpeechError
            If the microphone cannot be opened or a read error occurs.
        InteractionEnd
            If the user interrupts with Ctrl-C (``KeyboardInterrupt``),
            so the conversation ends gracefully -- consistent with
            ``TextInteraction``.
        """
        sd = self._get_sd()
        started = perf_counter()
        logger.info(
            "capture: STARTED (limit=%.2f s, silence=%.2f s, threshold=%.3f, "
            "chunk=%.2f ms)",
            self._recording_limit,
            self._silence_duration,
            self._silence_threshold,
            self._chunk_size / self._sample_rate * 1000,
        )

        max_chunks = max(
            1,
            int(self._recording_limit * self._sample_rate / self._chunk_size),
        )
        silence_chunks_needed = max(
            1,
            int(self._silence_duration * self._sample_rate / self._chunk_size),
        )

        frames: list[bytes] = []
        consecutive_silence = 0
        # Voice-activity gate: the utterance must actually START before any
        # silence is allowed to end it. Leading (pre-utterance) silence in the
        # moment before the user speaks must NEVER end the capture, otherwise a
        # short background gap ~= silence_duration finalizes ~0.3 s of noise and
        # Vosk returns an empty transcript (the reported real-voice failure).
        speech_started = False
        speech_start_sec = 0.0

        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
            )
        except Exception as exc:
            raise SpeechError(
                f"Cannot open microphone input stream: {exc}"
            ) from exc

        try:
            with stream:
                stream.start()
                for i in range(max_chunks):
                    try:
                        chunk, _overflow = stream.read(self._chunk_size)
                    except KeyboardInterrupt:
                        raise InteractionEnd from None
                    except Exception as exc:
                        raise SpeechError(
                            f"Microphone read failed: {exc}"
                        ) from exc

                    pcm = (
                        chunk.tobytes()
                        if hasattr(chunk, "tobytes")
                        else bytes(chunk)
                    )
                    frames.append(pcm)

                    elapsed_sec = (i + 1) * self._chunk_size / self._sample_rate
                    rms = _rms16(pcm)
                    if rms >= self._silence_threshold:
                        # Voiced energy. This either STARTS the utterance or
                        # resets the end-of-speech timer.
                        if not speech_started:
                            speech_started = True
                            speech_start_sec = elapsed_sec
                            logger.info(
                                "capture: speech detected at %.2f s",
                                elapsed_sec,
                            )
                        consecutive_silence = 0
                    else:
                        # Silent chunk.
                        if speech_started:
                            consecutive_silence += 1
                            if consecutive_silence >= silence_chunks_needed:
                                # End-of-speech: <silence_duration>s of trailing
                                # silence AFTER real speech. Includes the trailing
                                # quiet so the whole sentence is captured.
                                logger.info(
                                    "capture: END of utterance at %.2f s "
                                    "(trailing silence %.2f s, speech began "
                                    "%.2f s)",
                                    elapsed_sec,
                                    self._silence_duration,
                                    speech_start_sec,
                                )
                                break
                        # else: leading silence before any speech -- keep
                        # listening. It never ends the capture here; only the
                        # hard recording_limit bounds a silent wait.
        except InteractionEnd:
            raise
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(
                f"Microphone capture failed: {exc}"
            ) from exc

        pcm_data = b"".join(frames)
        if not pcm_data:
            raise SpeechError(
                "Microphone capture produced no audio data"
            )

        duration_s = len(pcm_data) / (self._sample_rate * self._channels * _SAMPLE_WIDTH)
        logger.info(
            "capture: done in %.0f ms, %.2f s of audio%s "
            "(speech begun at %.2f s)",
            (perf_counter() - started) * 1000,
            duration_s,
            " (no speech detected)" if not speech_started else "",
            speech_start_sec,
        )
        wav_data = _encode_wav(pcm_data, self._sample_rate, self._channels)
        return Audio(data=wav_data, format="wav")

    def playback(self, audio: Audio) -> None:
        """Play a WAV ``Audio`` payload through the default output device.

        Parameters
        ----------
        audio:
            An ``Audio`` whose ``data`` is a complete RIFF/WAVE container
            (e.g. produced by ``KokoroSpeechOutput.synthesize`` or
            ``StubSpeechOutput``).

        Raises
        ------
        SpeechError
            If the WAV cannot be decoded or the speaker fails.
        InteractionEnd
            If the user interrupts playback with Ctrl-C.
        """
        sd = self._get_sd()
        np = self._get_np()

        pcm, sample_rate, channels, sample_width = _decode_wav(
            bytes(audio.data)
        )

        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        np_dtype = dtype_map.get(sample_width, np.int16)
        # Use frombuffer for efficiency, but ensure the resulting array owns its
        # memory so it cannot be invalidated if the original bytes are freed.
        try:
            arr = np.frombuffer(pcm, dtype=np_dtype)
            # Ensure a real copy where supported so downstream playback always
            # references memory owned by the ndarray (avoids dangling buffers).
            try:
                arr = np.array(arr, copy=True)
            except Exception:
                # Some injected fake numpy modules in tests may not implement
                # array/copy semantics; fall back to leaving arr as-is.
                pass
        except Exception:
            # Some fake numpy modules might only implement minimal hooks; try a
            # best-effort conversion using the module's frombuffer alternative.
            try:
                arr = np.frombuffer(pcm, dtype=np_dtype)
            except Exception:
                # As last resort, create a Python bytes-backed object to play.
                arr = pcm

        if channels > 1:
            try:
                arr = arr.reshape(-1, channels)
            except Exception:
                pass

        # Kokoro synthesizes at its voice's native rate (e.g. 24 000 Hz). The output
        # device's native rate is often higher (e.g. 44 100 Hz on MME hosts), and
        # off-rate 22 050 Hz is serviced incorrectly -> playback at ~2x speed
        # (chipmunk "ki-ka-pa"). Resample to the device rate so the speech plays
        # at the intended pitch/duration.
        output_rate = self._resolve_output_sample_rate(sd, sample_rate)

        def _safe_metrics(np_mod: Any, a: Any) -> tuple[str, int, Any, Any, float]:
            """Return tuple (dtype_repr, samples, shape, minmax, nonzero_ratio) when possible.
            Best-effort: tests inject fake numpy which may not support operations; fall back to len()/repr()."""
            try:
                import numpy as _np  # local fallback for real numpy ops
                # prefer provided np_mod if it looks like numpy
                _np = np_mod if hasattr(np_mod, "any") or hasattr(np_mod, "interp") else _np
                arr_np = _np.asarray(a)
                dtype_repr = getattr(arr_np, "dtype", type(arr_np).__name__)
                samples = int(arr_np.shape[0]) if arr_np.size >= 0 else int(len(a))
                shape = getattr(arr_np, "shape", None)
                try:
                    mn = float(_np.min(arr_np))
                    mx = float(_np.max(arr_np))
                except Exception:
                    mn = None
                    mx = None
                try:
                    nonzero = float(_np.count_nonzero(arr_np)) / max(1, arr_np.size)
                except Exception:
                    nonzero = 0.0
                return str(dtype_repr), samples, shape, (mn, mx), nonzero
            except Exception:
                # best-effort fallback for fake np/arr used in unit tests
                try:
                    samples = len(a)
                except Exception:
                    samples = 0
                return type(a).__name__, samples, None, (None, None), 0.0

        # If resampling is needed, perform it and validate the result.
        if output_rate != sample_rate:
            n_before = len(arr)
            arr_before_metrics = _safe_metrics(np, arr)
            arr = self._resample_audio(np, arr, sample_rate, output_rate)
            arr_after_metrics = _safe_metrics(np, arr)
            logger.info(
                "playback: resampled audio %d -> %d Hz (%d -> %d samples) dtype=%s -> %s, minmax=%s -> %s, nonzero_ratio=%.3f -> %.3f",
                sample_rate,
                output_rate,
                n_before,
                arr_after_metrics[1],
                arr_before_metrics[0],
                arr_after_metrics[0],
                arr_before_metrics[3],
                arr_after_metrics[3],
                arr_before_metrics[4],
                arr_after_metrics[4],
            )
            # Validate the resampled array contains some non-zero data; warn if it's silent
            if arr_after_metrics[4] <= 0.0:
                logger.warning("playback: resampling produced silent audio (nonzero_ratio=0); will attempt playback but this may be audible silence)")
        else:
            metrics = _safe_metrics(np, arr)
            logger.info(
                "playback: %d Hz (matches device), %d samples, %d ch, dtype=%s, minmax=%s, nonzero_ratio=%.3f",
                sample_rate,
                metrics[1],
                channels,
                metrics[0],
                metrics[3],
                metrics[4],
            )

        # Gather device info (best-effort) to aid diagnosis on real systems.
        try:
            default_dev = getattr(sd, "default", None)
            query = getattr(sd, "query_devices", None)
            hostapi_q = getattr(sd, "query_hostapi", None)
            dev_index = None
            dev_info = None
            host_info = None
            if default_dev is not None:
                # sd.default.device may be a pair or single id
                try:
                    dev_index = default_dev.device if hasattr(default_dev, "device") else default_dev
                except Exception:
                    dev_index = default_dev
            if query is not None and dev_index is not None:
                try:
                    dev_info = query(device=int(dev_index) if not isinstance(dev_index, (list, tuple)) else int(dev_index[1] if len(dev_index) > 1 else dev_index[0]))
                except Exception:
                    try:
                        dev_info = query(device=dev_index)
                    except Exception:
                        dev_info = None
            if hostapi_q is not None and dev_info is not None:
                try:
                    host_info = hostapi_q(dev_info.get("hostapi") if isinstance(dev_info, dict) else None)
                except Exception:
                    host_info = None
            logger.info("playback: device_info=%s hostapi=%s", dev_info, host_info)
        except Exception:
            logger.exception("playback: failed to query device info")

        # Ensure samplerate is a plain int for backends picky about numpy scalar types
        sr = int(output_rate)

        # More detailed numeric metrics using the injected numpy module.
        try:
            arr_np = np.asarray(arr)
            # sample count and channels
            samples = int(arr_np.size) if hasattr(arr_np, "size") else len(arr)
            frames = int(arr_np.shape[0]) if hasattr(arr_np, "shape") else samples
            chs = int(arr_np.shape[1]) if getattr(arr_np, "ndim", 1) > 1 and len(arr_np.shape) > 1 else channels
            # min/max and RMS
            try:
                mn = float(arr_np.min())
                mx = float(arr_np.max())
                # RMS over int types: square mean then sqrt, normalized to full scale
                if arr_np.dtype.kind in "iu":
                    maxscale = float(2 ** (8 * arr_np.itemsize - 1))
                    rms = float(_np_rms(arr_np) / maxscale)
                else:
                    rms = float(_np_rms(arr_np))
            except Exception:
                mn = mx = rms = None
        except Exception:
            # Best-effort fallbacks for fake numpy in tests
            try:
                samples = len(arr)
            except Exception:
                samples = 0
            frames = samples
            chs = channels
            mn = mx = rms = None

        logger.info(
            "playback: pre-play metrics: wav_rate=%d wav_channels=%d arr_dtype=%s frames=%d channels=%d samples=%d min=%s max=%s rms=%s",
            sample_rate,
            channels,
            getattr(getattr(arr, "dtype", None), "name", type(arr).__name__),
            frames,
            chs,
            samples,
            mn,
            mx,
            rms,
        )

        # Attempt playback with conservative fallbacks and explicit logging.
        try:
            if self._barge_in:
                self._playback_bargeable(sd, arr, sr)
            else:
                # If an explicit sd_module was injected (tests), prefer its
                # OutputStream path to match test expectations. In real
                # runtime (no injected module) the primary path uses sd.play
                # which historically produced reliable continuous playback.
                # Decide whether to use OutputStream for long audio. Prefer sd.play
                # for short responses (historically reliable), but use OutputStream
                # for longer buffers to avoid handing massive single buffers to some
                # backends. Threshold chosen conservatively to match existing tests.
                try:
                    arr_duration = float(frames) / float(sr) if sr and frames else 0.0
                except Exception:
                    arr_duration = 0.0
                use_output_stream = getattr(sd, "OutputStream", None) is not None and arr_duration > 3.0
                if use_output_stream:
                    try:
                        arr_np = np.asarray(arr)
                        out_channels = arr_np.shape[1] if getattr(arr_np, "ndim", 1) > 1 and len(arr_np.shape) > 1 else channels
                        with sd.OutputStream(samplerate=sr, channels=out_channels, dtype=getattr(arr_np, 'dtype', None)) as stream:
                            stream.write(arr_np)
                        logger.info("playback: used OutputStream for long playback (duration=%.2fs)", arr_duration)
                        return
                    except Exception:
                        logger.exception("playback: OutputStream failed for long playback; falling back to sd.play")
                # Use sd.play as the first attempt — this was the previously working behavior
                try:
                    sd.play(arr, samplerate=sr, blocking=True)
                    wait_fn = getattr(sd, "wait", None)
                    if callable(wait_fn):
                        try:
                            wait_fn()
                        except Exception:
                            logger.exception("playback: sd.wait() failed after primary play")
                    # Primary path succeeded; return to caller to avoid executing fallback logic
                    return
                except Exception as exc_primary:
                    logger.exception("playback: primary sd.play failed using samplerate %s, attempting fallback: %s", sr, exc_primary)
                    OutputStream = getattr(sd, "OutputStream", None)
                    # First fallback: try the WAV's native rate
                    try:
                        logger.info("playback: attempting fallback to WAV sample rate %d", sample_rate)
                        sd.play(arr, samplerate=int(sample_rate), blocking=True)
                        wait_fn = getattr(sd, "wait", None)
                        if callable(wait_fn):
                            try:
                                wait_fn()
                            except Exception:
                                logger.exception("playback: sd.wait() failed after wav-fallback play")
                        logger.info("playback: fallback to WAV rate succeeded")
                        return
                    except Exception:
                        logger.exception("playback: fallback to WAV rate failed; attempting float32 conversion")
                    # Second fallback: convert integer PCM to float32 and retry
                    try:
                        arr_conv = _ensure_float32_for_playback(np, arr)
                        sd.play(arr_conv, samplerate=sr, blocking=True)
                        wait_fn = getattr(sd, "wait", None)
                        if callable(wait_fn):
                            try:
                                wait_fn()
                            except Exception:
                                logger.exception("playback: sd.wait() failed after float32 conversion play")
                        logger.info("playback: float32 conversion playback succeeded")
                        return
                    except Exception:
                        logger.exception("playback: float32 conversion playback failed")
                    # Final fallback: if OutputStream is available, attempt a simple single-write
                    if OutputStream is not None:
                        try:
                            arr_np = np.asarray(arr)
                            out_channels = arr_np.shape[1] if getattr(arr_np, "ndim", 1) > 1 and len(arr_np.shape) > 1 else channels
                            logger.info("playback: attempting OutputStream single-write fallback (samples=%d, sr=%d, channels=%d)", int(arr_np.shape[0]), sr, out_channels)
                            with OutputStream(samplerate=sr, channels=out_channels, dtype=getattr(arr_np, 'dtype', None)) as stream:
                                stream.write(arr_np)
                            logger.info("playback: OutputStream single-write completed")
                            return
                        except Exception as exc_out:
                            logger.exception("playback: OutputStream fallback failed: %s", exc_out)
                    # All fallbacks failed
                    raise SpeechError("Speaker playback failed after all fallbacks") from exc_primary
        except KeyboardInterrupt:
            raise InteractionEnd from None
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"Speaker playback failed: {exc}") from exc

    def _resolve_output_sample_rate(self, sd: Any, wav_rate: int) -> int:
        """Determine the sample rate to hand the output device.

        When ``output_sample_rate`` is explicitly set (e.g. via env var
        ``AURA_PLAYBACK_SAMPLE_RATE`` or constructor kwarg), that value
        wins.  Otherwise the output device's native ``default_samplerate``
        is queried (via ``sd.query_devices``) and used -- this is what
        prevents the "ki-ka-pa" chipmunk distortion when Kokoro's 24 000 Hz
        output is played on a 44 100 Hz MME device.  Any lookup failure
        gracefully falls back to the WAV's own rate so playback never
        crashes.
        """
        if self._output_sample_rate is not None:
            return int(self._output_sample_rate)

        try:
            default = getattr(sd, "default", None)
            query = getattr(sd, "query_devices", None)
            if default is None or query is None:
                return wav_rate
            device = getattr(default, "device", None)
            # ``sd.default.device`` may be a sounddevice ``_InputOutputPair``
            # (which supports indexing but NOT ``len()``), a numpy array, a
            # list/tuple ``[input_dev, output_dev]``, or an int.  Try to grab
            # the output device index from index 1; fall back to the whole
            # value if that fails (some backends return a single device id).
            output_dev = None
            if device is not None:
                try:
                    output_dev = device[1]
                except (IndexError, TypeError, KeyError):
                    output_dev = device
            if output_dev is None:
                return wav_rate
            # Force plain int for sounddevice's query_devices (some backends
            # are picky about numpy scalar types).
            info = query(device=int(output_dev))
            if not info:
                return wav_rate
            rate = float(info.get("default_samplerate") or wav_rate)
            return int(round(rate))
        except Exception:
            return wav_rate


    @staticmethod
    def _resample_audio(np: Any, arr: Any, from_rate: int, to_rate: int) -> Any:
        """Resample ``arr`` from ``from_rate`` to ``to_rate`` (numpy linear).

        Preserves the array dtype. Mono (1-D) and per-channel (2-D) both handled.
        Returns ``arr`` unchanged when the rates are equal or invalid.
        """
        if from_rate <= 0 or to_rate <= 0 or from_rate == to_rate:
            return arr

        def _one(a: Any) -> Any:
            n_out = max(1, int(round(a.shape[0] * to_rate / from_rate)))
            x_old = np.arange(a.shape[0], dtype=np.float64)
            x_new = np.linspace(0, a.shape[0] - 1, num=n_out)
            out = np.interp(x_new, x_old, a.astype(np.float64))
            if out.dtype != a.dtype:
                out = np.rint(out).astype(a.dtype)
            return out

        if arr.ndim == 1:
            return _one(arr)
        return np.stack([_one(arr[:, c]) for c in range(arr.shape[1])], axis=1)

    def _playback_bargeable(self, sd: Any, arr: Any, sample_rate: int) -> bool:
        """Play non-blocking while listening for user speech (single-threaded).

        Starts ``sd.play(..., blocking=False)`` and, in the SAME thread, reads
        the microphone in small chunks and runs the same RMS ``silence_threshold``
        VAD used by ``capture``. As soon as ``barge_sustain`` seconds of
        *consecutive* voiced audio is heard, playback is stopped and ``True`` is
        returned (the caller's ``write`` returns; ``Session`` then uses the
        normal ``capture`` for the new utterance). If the mic cannot be opened,
        or no speech is heard before playback finishes, ``False`` is returned.

        Single-threaded polling (no listener thread) so there are no races and
        no overlapping TTS: barge always stops the current reply before the next
        one can start.

        Returns
        -------
        bool
            True if playback was interrupted by detected user speech.
        """
        blocksize = max(256, self._chunk_size)
        confirm = max(1, int(self._barge_sustain * sample_rate / blocksize))
        logger.info(
            "barge-in: playing interruptibly (blocksize=%d, sustain=%.2fs -> %d chunks)",
            blocksize,
            self._barge_sustain,
            confirm,
        )

        interrupted = False
        mic = None
        try:
            mic = sd.InputStream(
                samplerate=sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=blocksize,
            )
        except Exception as exc:
            # Can't listen while speaking -- degrade to plain non-blocking
            # playback rather than crash AURA.
            logger.warning(
                "barge-in: could not open mic during playback (%s); playing without barge-in",
                exc,
            )
            self._play_non_blocking(sd)
            return False

        sd.play(arr, samplerate=sample_rate, blocking=False)
        try:
            with mic:
                consecutive = 0
                while True:
                    try:
                        chunk, _overflow = mic.read(blocksize)
                    except KeyboardInterrupt:
                        raise InteractionEnd from None
                    except Exception as exc:
                        raise SpeechError(f"Barge-in mic read failed: {exc}") from exc

                    pcm = (
                        chunk.tobytes()
                        if hasattr(chunk, "tobytes")
                        else bytes(chunk)
                    )
                    rms = _rms16(pcm)
                    if rms >= self._silence_threshold:
                        consecutive += 1
                        if consecutive >= confirm:
                            interrupted = True
                            break
                    else:
                        consecutive = 0

                    # Did the non-blocking playback finish on its own?
                    get_stream = getattr(sd, "get_stream", None)
                    if get_stream is not None:
                        stream = get_stream()
                        if stream is None:
                            break
                        active = getattr(stream, "active", True)
                        if not active:
                            break
        except InteractionEnd:
            raise
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"Barge-in playback failed: {exc}") from exc
        finally:
            # Drain/abort output and always tear it down so no stream lingers.
            if interrupted:
                logger.info("barge-in: user speech detected; interrupting playback")
            self._stop_playback(sd)

        return interrupted

    @staticmethod
    def _play_non_blocking(sd: Any) -> None:
        """Wait for and then stop a non-blocking playback (fallback)."""
        try:
            sd.wait()
        finally:
            try:
                sd.stop()
            except Exception:  # pragma: no cover - already shutting down
                pass

    @staticmethod
    def _stop_playback(sd: Any) -> None:
        """Abort/clean up any active playback without raising."""
        try:
            sd.stop()
        except Exception:  # pragma: no cover - already shutting down
            pass
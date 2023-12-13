import bisect
import functools
import os
import warnings

from typing import List, NamedTuple, Optional

import numpy as np
import auditok
import io
import scipy.io.wavfile

from faster_whisper.utils import get_assets_path


class VadOptions(NamedTuple):
    """VAD options.

    Attributes:
      energy_threshold: Spectral energy of audio threshold.
      min_speech_duration_s: Minimal speech chunks length.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Better set big.
      min_silence_duration_ms: Maximum duration of tolerated continuous silence within an event
      dialation_s: Padding audio chunk from each side
      mixing_distance_s: If speech chunks closer than this value, it will be merged
    """

    min_speech_duration_s: float = 0.2  # minimum duration of a valid audio event in seconds
    max_speech_duration_s: float = 36000 # maximum duration of an event (10 hours now)
    min_silence_duration_s: float = 0.5  # maximum duration of tolerated continuous silence within an event
    dialation_s: float = 1.
    mixing_distance_s: float = 3.
    energy_threshold: int = 60


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[VadOptions] = None,
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """

    SAMPLE_RATE = 16000
    audio_darution_s = audio.shape[0]/SAMPLE_RATE

    if audio_darution_s < 0.5:
        return []

    if vad_options is None:
        vad_options = VadOptions_1(max_speech_duration_s=audio_darution_s)

    # Cheap normalization of the volume
    audio = audio / max(0.1, np.max(np.abs(audio)))

    byte_io = io.BytesIO(bytes())
    scipy.io.wavfile.write(byte_io, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    bytes_wav = byte_io.read()

    segments = auditok.split(
        bytes_wav,
        sampling_rate=SAMPLE_RATE,                      # sampling frequency in Hz
        channels=1,                                     # number of channels
        sample_width=2,                                 # number of bytes per sample
        min_dur=vad_options.min_speech_duration_s,      # minimum duration of a valid audio event in seconds
        max_dur=vad_options.max_speech_duration_s,      # maximum duration of an event
        max_silence=vad_options.min_silence_duration_s, # maximum duration of tolerated continuous silence within an event
        energy_threshold=vad_options.energy_threshold,
        drop_trailing_silence=True,
    )

    segments = [{"start": s._meta.start * SAMPLE_RATE, "end": s._meta.end * SAMPLE_RATE} for s in segments]

    #Mixing nearest segments
    dif_thresh = vad_options.mixing_distance_s * SAMPLE_RATE
    speech_timestamps = segments
    mixed_timestamps = []
    if len(speech_timestamps) > 1:
        start_point = speech_timestamps[0]["start"]
        end_point = speech_timestamps[0]["end"]
        for i in range(len(speech_timestamps) - 1):
            if speech_timestamps[i + 1]["start"] - end_point < dif_thresh:
                end_point = speech_timestamps[i + 1]["end"]
            else:
                mixed_timestamps.append({"start": start_point, "end": end_point})
                start_point = speech_timestamps[i + 1]["start"]
                end_point = speech_timestamps[i + 1]["end"]

        mixed_timestamps.append({"start": start_point, "end": end_point})
        segments = mixed_timestamps

    if vad_options.dialation_s > 0:
        dilatation = round(vad_options.dialation_s * SAMPLE_RATE)
        new_segments = []
        for seg in segments:
            new_seg = {
                "start": max(0, seg["start"] - dilatation),
                "end": min(len(audio), seg["end"] + dilatation)
            }
            if len(new_segments) > 0 and new_segments[-1]["end"] >= new_seg["start"]:
                new_segments[-1]["end"] = new_seg["end"]
            else:
                new_segments.append(new_seg)
        segments = new_segments


    for seg in segments:
        seg["start"] = round(seg["start"])
        seg["end"] = round(seg["end"])

    sec_segments = [(el["start"]/SAMPLE_RATE, el["end"]/SAMPLE_RATE) for el in segments]
    print(sec_segments)
    return segments


def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
    """Collects and concatenates audio chunks."""
    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate([audio[chunk["start"] : chunk["end"]] for chunk in chunks])


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: List[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float) -> int:
        sample = int(time * self.sampling_rate)
        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )
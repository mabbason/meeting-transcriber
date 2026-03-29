"""
Speaker diarization using pyannote.audio.
Assigns speaker labels to transcription segments.
"""

import numpy as np
import torch

import config

DIARIZATION_AVAILABLE = False


class Diarizer:
    def __init__(self):
        self.pipeline = None
        self.speaker_map = {}  # Map pyannote labels to consistent Speaker N labels
        self.next_speaker_id = 1

    def load_model(self):
        global DIARIZATION_AVAILABLE

        if not config.HF_TOKEN:
            print("WARNING: HF_TOKEN not set — speaker diarization disabled")
            print("  Set HF_TOKEN in .env and accept terms at:")
            print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
            return

        try:
            from pyannote.audio import Pipeline

            print("Loading pyannote diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=config.HF_TOKEN,
            )

            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("Diarization pipeline loaded (CUDA)")
            else:
                print("Diarization pipeline loaded (CPU)")

            DIARIZATION_AVAILABLE = True
        except Exception as e:
            print(f"WARNING: Could not load diarization pipeline: {e}")
            print("  Speaker diarization will be disabled")

    def diarize(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        """
        Assign speaker labels to transcription segments.
        Modifies segments in-place, adding 'speaker' field.
        Returns the modified segments.
        """
        if self.pipeline is None:
            for seg in segments:
                seg["speaker"] = "Speaker"
            return segments

        try:
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": config.AUDIO_SAMPLE_RATE}
            )

            # Build timeline of speaker segments
            speaker_timeline = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "label": speaker_label,
                })

            # Assign speakers to transcription segments by overlap
            for seg in segments:
                seg["speaker"] = self._find_speaker(
                    seg["start"], seg["end"], speaker_timeline
                )

        except Exception as e:
            print(f"Diarization error: {e}")
            for seg in segments:
                seg["speaker"] = "Speaker"

        return segments

    def _find_speaker(self, seg_start, seg_end, timeline):
        """Find the speaker with most overlap for a given time range."""
        # Adjust to chunk-relative times for comparison
        best_overlap = 0
        best_label = None

        for turn in timeline:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn["label"]

        if best_label is None:
            return "Speaker"

        # Map pyannote's internal label to a consistent "Speaker N"
        if best_label not in self.speaker_map:
            self.speaker_map[best_label] = f"Speaker {self.next_speaker_id}"
            self.next_speaker_id += 1

        return self.speaker_map[best_label]

    def reset(self):
        self.speaker_map = {}
        self.next_speaker_id = 1

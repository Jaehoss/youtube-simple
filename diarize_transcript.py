"""
Speaker Diarization + Whisper Transcription for Apple Silicon (M4 Pro)

Uses:
  - mlx-whisper for GPU-accelerated transcription on Apple Silicon
  - pyannote-audio for speaker diarization

Setup:
  pip install mlx-whisper pyannote.audio

  You MUST have a HuggingFace token with access to:
    - https://huggingface.co/pyannote/segmentation-3.0
    - https://huggingface.co/pyannote/speaker-diarization-3.1
  Accept the user agreements on both model pages, then set:
    export HF_TOKEN="hf_your_token_here"
"""

import os
import sys
import json
from pathlib import Path

import mlx_whisper
from pyannote.audio import Pipeline as DiarizationPipeline


def transcribe_with_whisper(audio_path: str, model: str = "mlx-community/whisper-large-v3-turbo") -> dict:
    """Run mlx-whisper transcription with word-level timestamps."""
    print(f"Transcribing with mlx-whisper ({model})...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        word_timestamps=True,
    )
    return result


def diarize(audio_path: str, hf_token: str, num_speakers: int | None = None) -> list[dict]:
    """Run pyannote speaker diarization. Returns list of {start, end, speaker}."""
    print("Running speaker diarization with pyannote...")
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # pyannote uses PyTorch; on Apple Silicon it can use MPS for some ops
    # Uncomment the next two lines to try MPS acceleration (may cause warnings):
    # import torch
    # pipeline.to(torch.device("mps"))

    diarization_kwargs = {}
    if num_speakers is not None:
        diarization_kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **diarization_kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return segments


def assign_speakers_to_segments(whisper_result: dict, diarization_segments: list[dict]) -> list[dict]:
    """
    Merge Whisper segments with diarization labels.
    For each Whisper segment, find the diarization speaker with the most overlap.
    """
    output = []
    for seg in whisper_result.get("segments", []):
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()

        # Find overlapping diarization segments and pick the one with most overlap
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for d in diarization_segments:
            overlap_start = max(seg_start, d["start"])
            overlap_end = min(seg_end, d["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        output.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": best_speaker,
            "text": seg_text,
        })
    return output


def format_transcript(labeled_segments: list[dict]) -> str:
    """Format labeled segments into a readable transcript."""
    lines = []
    current_speaker = None
    for seg in labeled_segments:
        if seg["speaker"] != current_speaker:
            current_speaker = seg["speaker"]
            timestamp = f"[{seg['start']:.1f}s]"
            lines.append(f"\n{current_speaker} {timestamp}:")
        lines.append(f"  {seg['text']}")
    return "\n".join(lines).strip()


def diarize_transcript(
    audio_path: str,
    hf_token: str | None = None,
    whisper_model: str = "mlx-community/whisper-large-v3-turbo",
    num_speakers: int | None = None,
    output_json: str | None = None,
) -> str:
    """
    Full pipeline: transcribe with mlx-whisper + diarize with pyannote.
    Returns formatted transcript string.
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass
    if not hf_token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass hf_token argument.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Then accept agreements at:\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    # Step 1: Transcribe
    whisper_result = transcribe_with_whisper(audio_path, model=whisper_model)

    # Step 2: Diarize
    diarization_segments = diarize(audio_path, hf_token, num_speakers=num_speakers)

    # Step 3: Merge
    labeled = assign_speakers_to_segments(whisper_result, diarization_segments)

    # Step 4: Optionally save JSON
    if output_json:
        with open(output_json, "w") as f:
            json.dump(labeled, f, indent=2, ensure_ascii=False)
        print(f"Saved labeled JSON to {output_json}")

    # Step 5: Format
    transcript = format_transcript(labeled)
    return transcript


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diarize_transcript.py <audio_file> [num_speakers]")
        print("  Set HF_TOKEN env var before running.")
        sys.exit(1)

    audio_file = sys.argv[1]
    n_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    result = diarize_transcript(
        audio_path=audio_file,
        num_speakers=n_speakers,
        output_json=Path(audio_file).stem + "_diarized.json",
    )
    print("\n" + "=" * 60)
    print(result)

"""Toggleable developer diagnostics panel (Step 4.4, §138)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from jammate.analysis.notes import midi_to_name
from jammate.audio.levels import SILENCE_FLOOR_DBFS

if TYPE_CHECKING:
    from jammate.session import JamSession


@dataclass(frozen=True, slots=True)
class DebugDiagnostics:
    """Snapshot of developer-mode diagnostics polled on the GUI thread (§138)."""

    rms_dbfs: float = SILENCE_FLOOR_DBFS
    peak_dbfs: float = SILENCE_FLOOR_DBFS
    pitch_hz: float | None = None
    note_name: str = "--"
    pitch_confidence: float = 0.0
    bpm: float | None = None
    tempo_confidence: float = 0.0
    tempo_locked: bool = False
    beat: int | None = None
    bar: int | None = None
    beat_phase: float | None = None
    intensity: float = 0.0
    drum_state: str = "stopped"
    drum_pattern: str = "--"
    ai_status: str = "disabled"
    ai_buffer_seconds: float = 0.0
    ai_ms_per_frame: float = 0.0
    input_overflows: int = 0
    output_underflows: int = 0
    ai_underruns: int = 0

    @classmethod
    def from_session(cls, session: JamSession | None) -> DebugDiagnostics:
        """Build a non-blocking diagnostics snapshot from `session` (or defaults when `None`)."""
        if session is None:
            return cls()
        state = session.state
        engine_stats = session.engine.stats()
        analysis_stats = session.worker.stats()
        ai_stats = session.ai.stats()

        latest = analysis_stats.latest
        rms_dbfs = SILENCE_FLOOR_DBFS if latest is None else latest.rms_dbfs
        pitch_hz = None if latest is None else latest.frequency_hz
        pitch_conf = 0.0 if latest is None else latest.confidence
        note_name = "--" if state.current_note is None else midi_to_name(state.current_note)

        drum_pattern = "--"
        if state.bar is not None:
            pat = session.drums.controller.pattern_for_bar(state.bar)
            if pat is not None:
                drum_pattern = pat.name
        if drum_pattern == "--" and session.drums.is_running:
            drum_pattern = f"{session.drums.style}_{session.drums.tier}"

        return cls(
            rms_dbfs=rms_dbfs,
            peak_dbfs=analysis_stats.recent_peak_dbfs,
            pitch_hz=pitch_hz,
            note_name=note_name,
            pitch_confidence=pitch_conf,
            bpm=state.bpm,
            tempo_confidence=state.tempo_confidence,
            tempo_locked=state.tempo_locked,
            beat=state.beat,
            bar=state.bar,
            beat_phase=state.beat_phase,
            intensity=state.intensity,
            drum_state=session.drums.drummer_state.value,
            drum_pattern=drum_pattern,
            ai_status=ai_stats.status.value,
            ai_buffer_seconds=ai_stats.buffer_seconds,
            ai_ms_per_frame=ai_stats.ms_per_frame,
            input_overflows=engine_stats.input_overflows,
            output_underflows=engine_stats.output_underflows,
            ai_underruns=engine_stats.ai_underruns,
        )


class DebugPanel(QGroupBox):
    """Collapsible developer panel exposing internal DSP, drum, AI, and xrun metrics (§138)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Developer Diagnostics", parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.rms_label = QLabel()
        self.pitch_label = QLabel()
        self.tempo_label = QLabel()
        self.beat_label = QLabel()
        self.intensity_label = QLabel()
        self.drum_pattern_label = QLabel()
        self.ai_label = QLabel()
        self.xruns_label = QLabel()

        layout.addRow("Input RMS:", self.rms_label)
        layout.addRow("Pitch:", self.pitch_label)
        layout.addRow("Tempo:", self.tempo_label)
        layout.addRow("Beat / Bar:", self.beat_label)
        layout.addRow("Intensity:", self.intensity_label)
        layout.addRow("Drum Pattern:", self.drum_pattern_label)
        layout.addRow("AI Engine:", self.ai_label)
        layout.addRow("Audio Xruns:", self.xruns_label)

        self.update_diagnostics(DebugDiagnostics())

    def update_diagnostics(self, diag: DebugDiagnostics) -> None:
        """Refresh all diagnostic labels on the GUI thread."""
        self.rms_label.setText(f"{diag.rms_dbfs:.1f} dBFS (peak {diag.peak_dbfs:.1f} dBFS)")

        if diag.pitch_hz is not None:
            self.pitch_label.setText(
                f"{diag.note_name} ({diag.pitch_hz:.1f} Hz, conf {diag.pitch_confidence:.2f})"
            )
        else:
            self.pitch_label.setText(f"{diag.note_name} (-- Hz, conf {diag.pitch_confidence:.2f})")

        if diag.bpm is not None:
            locked_str = "yes" if diag.tempo_locked else "no"
            self.tempo_label.setText(
                f"{diag.bpm:.1f} BPM (conf {diag.tempo_confidence:.2f}, locked: {locked_str})"
            )
        else:
            self.tempo_label.setText("-- BPM (conf 0.00, locked: no)")

        if diag.beat is not None and diag.bar is not None and diag.beat_phase is not None:
            self.beat_label.setText(
                f"beat {diag.beat}/4, bar {diag.bar}, phase {diag.beat_phase:.2f}"
            )
        else:
            self.beat_label.setText("-- / 4, bar --, phase --")

        self.intensity_label.setText(f"{diag.intensity:.2f}")
        self.drum_pattern_label.setText(f"{diag.drum_pattern} ({diag.drum_state})")
        self.ai_label.setText(
            f"{diag.ai_status} | buf {diag.ai_buffer_seconds * 1000.0:.0f} ms | "
            f"{diag.ai_ms_per_frame:.1f} ms/frame"
        )
        total_xruns = diag.input_overflows + diag.output_underflows
        self.xruns_label.setText(
            f"{total_xruns} (in {diag.input_overflows}, out {diag.output_underflows}, "
            f"ai {diag.ai_underruns})"
        )

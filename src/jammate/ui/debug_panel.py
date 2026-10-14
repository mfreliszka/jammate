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

    instrument: str = "bass"
    rms_dbfs: float = SILENCE_FLOOR_DBFS
    peak_dbfs: float = SILENCE_FLOOR_DBFS
    pitch_hz: float | None = None
    note_name: str = "--"
    chord_type: str | None = None
    pitch_confidence: float = 0.0
    estimated_key: str | None = None
    estimated_mode: str | None = None
    harmonic_confidence: float = 0.0
    bass_register: str = "low"
    riff_id: str | None = None
    riff_iteration: int = 0
    riff_changed: bool = False
    section: str = "listening"
    session_memory_count: int = 0
    bpm: float | None = None
    tempo_confidence: float = 0.0
    tempo_locked: bool = False
    beat: int | None = None
    bar: int | None = None
    beat_phase: float | None = None
    intensity: float = 0.0
    drum_state: str = "stopped"
    drum_pattern: str = "--"
    drum_role: str = "silent"
    ai_status: str = "disabled"
    ai_role: str = "silent"
    ai_model_size: str = "mrt2_small"
    ai_assets_summary: str = "unchecked"
    ai_buffer_seconds: float = 0.0
    ai_ms_per_frame: float = 0.0
    midi_status: str = "disabled"
    midi_port: str = "--"
    midi_active_notes: int = 0
    input_overflows: int = 0
    output_underflows: int = 0
    ai_underruns: int = 0

    @classmethod
    def from_session(cls, session: JamSession | None) -> DebugDiagnostics:
        """Build a non-blocking diagnostics snapshot from `session` (or defaults when `None`)."""
        if session is None:
            return cls()
        state = session.state
        directive = session.directive
        engine_stats = session.engine.stats()
        analysis_stats = session.worker.stats()
        ai_stats = session.ai.stats()
        midi_worker = getattr(session, "midi_worker", None)
        midi_stats = midi_worker.stats() if midi_worker is not None else None

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
            instrument=session.instrument,
            rms_dbfs=rms_dbfs,
            peak_dbfs=analysis_stats.recent_peak_dbfs,
            pitch_hz=pitch_hz,
            note_name=note_name,
            chord_type=state.chord_type,
            pitch_confidence=pitch_conf,
            estimated_key=state.estimated_key,
            estimated_mode=state.estimated_mode,
            harmonic_confidence=state.harmonic_confidence,
            bass_register=state.bass_register,
            riff_id=state.riff_id,
            riff_iteration=state.riff_iteration,
            riff_changed=state.riff_changed,
            section=state.section,
            session_memory_count=len(session.riff_detector.memory),
            bpm=state.bpm,
            tempo_confidence=state.tempo_confidence,
            tempo_locked=state.tempo_locked,
            beat=state.beat,
            bar=state.bar,
            beat_phase=state.beat_phase,
            intensity=state.intensity,
            drum_state=session.drums.drummer_state.value,
            drum_pattern=drum_pattern,
            drum_role=directive.drum_role,
            ai_status=ai_stats.status.value,
            ai_role=directive.ai_role,
            ai_model_size=session.config.ai.model_size,
            ai_buffer_seconds=ai_stats.buffer_seconds,
            ai_ms_per_frame=ai_stats.ms_per_frame,
            midi_status=midi_stats.status.value if midi_stats is not None else "disabled",
            midi_port=(
                midi_stats.port_name
                if (midi_stats is not None and midi_stats.port_name is not None)
                else "--"
            ),
            midi_active_notes=len(midi_stats.active_notes) if midi_stats is not None else 0,
            input_overflows=engine_stats.input_overflows,
            output_underflows=engine_stats.output_underflows,
            ai_underruns=engine_stats.ai_underruns,
        )


class DebugPanel(QGroupBox):
    """Collapsible developer panel exposing DSP, drum, AI, MIDI, and xrun metrics (§138)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Developer Diagnostics", parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.instrument_label = QLabel()
        self.midi_label = QLabel()
        self.rms_label = QLabel()
        self.pitch_label = QLabel()
        self.key_label = QLabel()
        self.riff_label = QLabel()
        self.tempo_label = QLabel()
        self.beat_label = QLabel()
        self.intensity_label = QLabel()
        self.drum_pattern_label = QLabel()
        self.ai_model_label = QLabel()
        self.model_assets_label = QLabel()
        self.ai_label = QLabel()
        self.xruns_label = QLabel()

        layout.addRow("Instrument:", self.instrument_label)
        layout.addRow("MIDI Input:", self.midi_label)
        layout.addRow("Input RMS:", self.rms_label)
        layout.addRow("Pitch:", self.pitch_label)
        layout.addRow("Key / Register:", self.key_label)
        layout.addRow("Riff / Memory:", self.riff_label)
        layout.addRow("Tempo:", self.tempo_label)
        layout.addRow("Beat / Bar:", self.beat_label)
        layout.addRow("Intensity:", self.intensity_label)
        layout.addRow("Drum Pattern:", self.drum_pattern_label)
        layout.addRow("AI Model:", self.ai_model_label)
        layout.addRow("Model Assets:", self.model_assets_label)
        layout.addRow("AI Engine:", self.ai_label)
        layout.addRow("Audio Xruns:", self.xruns_label)

        self.update_diagnostics(DebugDiagnostics())

    def update_diagnostics(self, diag: DebugDiagnostics) -> None:
        """Refresh all diagnostic labels on the GUI thread."""
        if diag.chord_type is not None:
            self.instrument_label.setText(f"{diag.instrument} ({diag.chord_type})")
        else:
            self.instrument_label.setText(diag.instrument)
        if diag.midi_status == "disabled":
            self.midi_label.setText("disabled")
        else:
            self.midi_label.setText(
                f"{diag.midi_port} ({diag.midi_status}, {diag.midi_active_notes} active)"
            )
        self.rms_label.setText(f"{diag.rms_dbfs:.1f} dBFS (peak {diag.peak_dbfs:.1f} dBFS)")

        if diag.pitch_hz is not None:
            self.pitch_label.setText(
                f"{diag.note_name} ({diag.pitch_hz:.1f} Hz, conf {diag.pitch_confidence:.2f})"
            )
        else:
            self.pitch_label.setText(f"{diag.note_name} (-- Hz, conf {diag.pitch_confidence:.2f})")

        if diag.estimated_key is not None:
            mode_str = f" {diag.estimated_mode}" if diag.estimated_mode else ""
            self.key_label.setText(
                f"{diag.estimated_key}{mode_str} (conf {diag.harmonic_confidence:.2f}, "
                f"reg {diag.bass_register})"
            )
        else:
            self.key_label.setText(f"-- (conf 0.00, reg {diag.bass_register})")

        if diag.riff_id is not None:
            changed_str = " [changed]" if diag.riff_changed else ""
            self.riff_label.setText(
                f"{diag.riff_id} x{diag.riff_iteration}{changed_str} "
                f"({diag.section}, mem {diag.session_memory_count})"
            )
        else:
            self.riff_label.setText(f"-- ({diag.section}, mem {diag.session_memory_count})")

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
        self.ai_model_label.setText(diag.ai_model_size)
        self.model_assets_label.setText(diag.ai_assets_summary)
        self.ai_label.setText(
            f"{diag.ai_status} | buf {diag.ai_buffer_seconds * 1000.0:.0f} ms | "
            f"{diag.ai_ms_per_frame:.1f} ms/frame"
        )
        total_xruns = diag.input_overflows + diag.output_underflows
        self.xruns_label.setText(
            f"{total_xruns} (in {diag.input_overflows}, out {diag.output_underflows}, "
            f"ai {diag.ai_underruns})"
        )

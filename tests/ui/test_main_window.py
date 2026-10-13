"""Tests for MainWindow and DebugPanel (Step 4.4)."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from tests.fakes import INPUT_DEVICE, OUTPUT_DEVICE, make_device

from jammate.ai.null import NullAccompaniment
from jammate.audio.devices import DeviceInventory
from jammate.audio.engine import AudioEngine
from jammate.config import AccompanimentConfig, JamMateConfig, TempoConfig
from jammate.music.events import PitchEvent, TempoUpdate
from jammate.session import JamSession
from jammate.ui.debug_panel import DebugDiagnostics, DebugPanel
from jammate.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


def _fake_inventory() -> DeviceInventory:
    return DeviceInventory(
        devices=(
            make_device(0, "Built-in Microphone", inputs=1, outputs=0),
            make_device(1, "Built-in Output", inputs=0, outputs=2),
            make_device(2, "USB Audio CODEC", inputs=2, outputs=2),
        ),
        default_input_index=0,
        default_output_index=1,
    )


def _null_session_factory(config: JamMateConfig) -> JamSession:
    return JamSession(
        config,
        engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
        accompaniment_factory=lambda ai_cfg, _sr: NullAccompaniment(
            style=ai_cfg.style,
            follow_improvise=ai_cfg.follow_improvise,
        ),
    )


def test_main_window_controls_full_jam_lifecycle_and_closes_cleanly(
    qapp: QApplication,
    fake_stream,
) -> None:
    del qapp, fake_stream
    cfg = JamMateConfig()
    window = MainWindow(
        cfg,
        session_factory=_null_session_factory,
        inventory_provider=_fake_inventory,
        preload_ai=False,
        shutdown_mlx_on_close=False,
    )
    window.show()

    assert window.status_label.text() == "Ready"
    assert window.jam_button.text() == "START JAM"
    assert window.note_label.text() == "--"
    assert window.bpm_label.text() == "-- BPM"
    assert [dot.text() for dot in window.beat_dots] == ["o 1", "o 2", "o 3", "o 4"]
    assert not window.debug_panel.isVisible()

    # Select USB Audio CODEC and start jamming
    usb_in_idx = window.input_combo.findData("USB Audio CODEC")
    usb_out_idx = window.output_combo.findData("USB Audio CODEC")
    assert usb_in_idx >= 0
    assert usb_out_idx >= 0
    window.input_combo.setCurrentIndex(usb_in_idx)
    window.output_combo.setCurrentIndex(usb_out_idx)

    assert window.start_jam()
    assert window.is_jamming
    assert window.jam_button.text() == "STOP JAM"
    assert not window.input_combo.isEnabled()
    assert not window.output_combo.isEnabled()
    assert window.status_label.text() == "Listening..."

    session = window.session
    assert session is not None
    assert session.config.audio.input_device_name == "USB Audio CODEC"
    assert session.config.audio.output_device_name == "USB Audio CODEC"

    # Feed events into the session and poll the UI
    session._handle_event(
        PitchEvent(
            frame_index=4_800,
            timestamp_seconds=0.1,
            midi_note=40,
            frequency_hz=82.41,
            confidence=0.95,
        )
    )
    session._handle_event(
        TempoUpdate(
            frame_index=9_600,
            timestamp_seconds=0.2,
            bpm=100.0,
            confidence=0.88,
            locked=True,
        )
    )
    window.poll_state()
    assert window.note_label.text() == "E2"
    assert window.bpm_label.text() == "100 BPM"
    assert any(dot.text().startswith("O ") for dot in window.beat_dots)

    # Toggle style, Drums ON/OFF + volume, AI ON/OFF + volume, Follow<->Improvise, Debug panel
    stoner_idx = window.style_combo.findData("stoner_rock")
    window.style_combo.setCurrentIndex(stoner_idx)
    assert session.style == "stoner_rock"

    window.drums_toggle.setChecked(False)
    assert window.drums_toggle.text() == "OFF"
    assert session.engine.processor.drums_muted is True
    window.drums_volume_slider.setValue(120)
    assert session.engine.processor.drums_gain == pytest.approx(1.2)

    window.ai_toggle.setChecked(False)
    assert window.ai_toggle.text() == "OFF"
    assert session.engine.processor.ai_muted is True
    window.ai_volume_slider.setValue(90)
    assert session.engine.processor.ai_gain == pytest.approx(0.9)

    window.follow_improvise_slider.setValue(80)
    assert session.follow_improvise == pytest.approx(0.8)

    window.debug_toggle.setChecked(True)
    assert window.debug_panel.isVisible()
    window.poll_state()
    assert "100.0 BPM" in window.debug_panel.tempo_label.text()

    # Closing the window while jamming stops the session and leaves no worker threads
    window.close()
    assert not window.is_jamming
    assert not session.is_running
    assert session.note_tracker.active_note is None
    jammate_threads = [
        t for t in threading.enumerate() if t.name.startswith("jammate-") and t.is_alive()
    ]
    assert jammate_threads == []


def test_main_window_reports_background_warmup_and_unavailable_ai(
    qapp: QApplication,
    fake_stream,
) -> None:
    del qapp, fake_stream
    gate = threading.Event()

    def slow_failing_warmup(_size: str) -> None:
        gate.wait(timeout=2.0)
        raise RuntimeError("weights missing")

    window = MainWindow(
        JamMateConfig(
            analysis={"tempo": TempoConfig(manual_bpm=100.0)},
            ai=AccompanimentConfig(enabled=True),
        ),
        session_factory=lambda cfg: JamSession(
            cfg,
            engine_factory=lambda audio: AudioEngine(audio, INPUT_DEVICE, OUTPUT_DEVICE),
            accompaniment_factory=lambda ai_cfg, _sr: NullAccompaniment(
                unavailable_reason="weights missing",
                style=ai_cfg.style,
                follow_improvise=ai_cfg.follow_improvise,
            ),
        ),
        inventory_provider=_fake_inventory,
        warmup_fn=slow_failing_warmup,
        preload_ai=True,
        shutdown_mlx_on_close=False,
    )
    window.show()
    try:
        assert window.status_label.text() == "Loading AI model..."
        assert window.warmup_progress.isVisible()

        gate.set()
        if window._warmup_thread is not None:
            window._warmup_thread.join(timeout=2.0)
        window.poll_state()
        assert window.status_label.text() == "Ready (AI unavailable)"
        assert not window.warmup_progress.isVisible()

        assert window.start_jam()
        window.poll_state()
        assert "AI unavailable" in window.status_label.text()
    finally:
        gate.set()
        window.close()


def test_debug_panel_formats_default_and_populated_diagnostics(qapp: QApplication) -> None:
    del qapp
    panel = DebugPanel()
    assert "-- Hz" in panel.pitch_label.text()
    assert "-- BPM" in panel.tempo_label.text()

    panel.update_diagnostics(
        DebugDiagnostics(
            rms_dbfs=-18.4,
            peak_dbfs=-6.2,
            pitch_hz=82.4,
            note_name="E2",
            pitch_confidence=0.92,
            bpm=92.5,
            tempo_confidence=0.85,
            tempo_locked=True,
            beat=3,
            bar=4,
            beat_phase=0.61,
            intensity=0.73,
            drum_state="playing",
            drum_pattern="stoner_rock_mid",
            ai_status="running",
            ai_buffer_seconds=0.32,
            ai_ms_per_frame=24.1,
            input_overflows=0,
            output_underflows=1,
            ai_underruns=2,
        )
    )
    assert panel.pitch_label.text() == "E2 (82.4 Hz, conf 0.92)"
    assert panel.tempo_label.text() == "92.5 BPM (conf 0.85, locked: yes)"
    assert panel.beat_label.text() == "beat 3/4, bar 4, phase 0.61"
    assert panel.drum_pattern_label.text() == "stoner_rock_mid (playing)"
    assert panel.ai_label.text() == "running | buf 320 ms | 24.1 ms/frame"
    assert panel.xruns_label.text() == "1 (in 0, out 1, ai 2)"


def test_main_window_record_toggle_and_preferences_persistence(
    qapp: QApplication,
    fake_stream,
    tmp_path,
) -> None:
    del qapp, fake_stream
    import numpy as np
    import soundfile as sf

    from jammate.config import RecordingConfig, load_preferences

    prefs_path = tmp_path / "settings.json"
    recordings_dir = tmp_path / "recordings"
    base_cfg = JamMateConfig(recording=RecordingConfig(output_dir=recordings_dir))

    window1 = MainWindow(
        base_cfg,
        session_factory=_null_session_factory,
        inventory_provider=_fake_inventory,
        preload_ai=False,
        shutdown_mlx_on_close=False,
        preferences_path=prefs_path,
    )
    window1.show()
    try:
        # Change devices, style, and volumes; verify preferences save on change
        usb_in = window1.input_combo.findData("USB Audio CODEC")
        usb_out = window1.output_combo.findData("USB Audio CODEC")
        window1.input_combo.setCurrentIndex(usb_in)
        window1.output_combo.setCurrentIndex(usb_out)
        window1.style_combo.setCurrentIndex(window1.style_combo.findData("funk"))
        window1.drums_volume_slider.setValue(110)
        window1.ai_volume_slider.setValue(60)

        saved = load_preferences(prefs_path)
        assert saved.input_device_name == "USB Audio CODEC"
        assert saved.output_device_name == "USB Audio CODEC"
        assert saved.style == "funk"
        assert saved.drums_gain == pytest.approx(1.1)
        assert saved.ai_gain == pytest.approx(0.6)

        # Toggle recording on during a jam and feed a callback block
        assert window1.record_toggle.text() == "Record"
        assert window1.start_jam()
        session = window1.session
        assert session is not None
        assert not session.is_recording

        window1.record_toggle.setChecked(True)
        assert window1.record_toggle.text() == "Recording"
        assert session.is_recording
        rec_path = session.recorder.file_path
        assert rec_path is not None

        indata = np.full((128, 1), 0.25, dtype=np.float32)
        outdata = np.zeros((128, 2), dtype=np.float32)
        session.engine.processor.process(indata, outdata)
    finally:
        window1.close()
    assert rec_path.is_file()
    info = sf.info(rec_path)
    assert info.frames == 128
    assert info.channels == 2

    # A second MainWindow launched with the same preferences_path restores all saved settings
    window2 = MainWindow(
        JamMateConfig(),
        session_factory=_null_session_factory,
        inventory_provider=_fake_inventory,
        preload_ai=False,
        shutdown_mlx_on_close=False,
        preferences_path=prefs_path,
    )
    try:
        assert window2.selected_input_device_name == "USB Audio CODEC"
        assert window2.selected_output_device_name == "USB Audio CODEC"
        assert window2.selected_style == "funk"
        assert window2.drums_gain == pytest.approx(1.1)
        assert window2.ai_gain == pytest.approx(0.6)
    finally:
        window2.close()

"""Main product window for JamMate (Step 4.4, §§56, 57, 83, 84, 138, 139)."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from jammate.ai.base import AccompanimentStatus
from jammate.ai.mrt import default_mrt2_system_factory, shutdown_mlx_runtime
from jammate.analysis.notes import midi_to_name
from jammate.audio.devices import DeviceInventory, query_inventory
from jammate.config import (
    DEFAULT_PREFERENCES_PATH,
    JamMateConfig,
    StyleFamily,
    UserPreferences,
    load_preferences,
    save_preferences,
)
from jammate.drums.engine import DrummerState
from jammate.music.events import PlayingState
from jammate.music.state import MusicalState
from jammate.music.styles import STYLE_PRESETS
from jammate.session import JamSession
from jammate.ui.debug_panel import DebugDiagnostics, DebugPanel

logger = logging.getLogger(__name__)

UI_POLL_INTERVAL_MS = 33  # ~30 Hz (§56, Step 4.4)
SLIDER_SCALE = 100  # 0..200 slider -> 0.0..2.0 gain; 0..100 slider -> 0.0..1.0 follow/improvise

type SessionFactory = Callable[[JamMateConfig], JamSession]
type InventoryProvider = Callable[[], DeviceInventory]
type ModelWarmupFn = Callable[[str], Any]


def format_plain_status(
    *,
    session: JamSession | None,
    warming_up: bool,
    warmup_failed: bool,
    last_error: str | None,
) -> str:
    """Derive the musician-facing plain-language status string (§139)."""
    if last_error is not None:
        return f"Audio error: {last_error}"
    if session is None or not session.is_running:
        if warming_up:
            return "Loading AI model..."
        if warmup_failed:
            return "Ready (AI unavailable)"
        return "Ready"

    state = session.state
    ai_status = session.ai.status
    ai_unavail_suffix = (
        " (AI unavailable)"
        if (
            not session.engine.processor.ai_muted
            and ai_status in (AccompanimentStatus.UNAVAILABLE, AccompanimentStatus.FAILED)
        )
        else ""
    )
    drummer_state = session.drums.drummer_state
    if drummer_state in (DrummerState.PLAYING, DrummerState.HOLDING):
        return f"Playing{ai_unavail_suffix}"
    if drummer_state == DrummerState.WAITING_FOR_BAR:
        return f"Joining on next bar...{ai_unavail_suffix}"
    if state.playing_state in (PlayingState.PLAYING, PlayingState.SUSTAINING):
        return f"Listening...{ai_unavail_suffix}"
    return f"Listening...{ai_unavail_suffix}"


class MainWindow(QMainWindow):
    """Simple musician-facing jam window with toggleable developer diagnostics (§56, §138, §139)."""

    def __init__(
        self,
        config: JamMateConfig | None = None,
        *,
        session_factory: SessionFactory = JamSession,
        inventory_provider: InventoryProvider = query_inventory,
        warmup_fn: ModelWarmupFn | None = default_mrt2_system_factory,
        preload_ai: bool = True,
        shutdown_mlx_on_close: bool = True,
        preferences_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        base_config = config or JamMateConfig()
        self._preferences_path = preferences_path
        if preferences_path is not None and preferences_path.is_file():
            prefs = load_preferences(preferences_path)
            base_config = prefs.apply_to_config(base_config)
        self._base_config = base_config
        self._session_factory = session_factory
        self._inventory_provider = inventory_provider
        self._warmup_fn = warmup_fn
        self._shutdown_mlx_on_close = shutdown_mlx_on_close

        self._session: JamSession | None = None
        self._last_error: str | None = None
        self._warming_up = False
        self._warmup_failed = False
        self._warmup_thread: threading.Thread | None = None

        self.setWindowTitle("JamMate")
        self.setMinimumWidth(420)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Title + plain-language status header (§56, §139)
        header_layout = QHBoxLayout()
        title_label = QLabel("JAMMATE")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_label)
        root_layout.addLayout(header_layout)

        # Model warmup progress bar (visible only during background model warmup)
        self.warmup_progress = QProgressBar()
        self.warmup_progress.setRange(0, 0)
        self.warmup_progress.setTextVisible(False)
        self.warmup_progress.setFixedHeight(6)
        self.warmup_progress.setVisible(False)
        root_layout.addWidget(self.warmup_progress)

        # Device selectors
        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(10)
        device_grid.setVerticalSpacing(6)
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        device_grid.addWidget(QLabel("Input"), 0, 0)
        device_grid.addWidget(self.input_combo, 0, 1)
        device_grid.addWidget(QLabel("Output"), 1, 0)
        device_grid.addWidget(self.output_combo, 1, 1)
        device_grid.setColumnStretch(1, 1)
        root_layout.addLayout(device_grid)
        self._populate_devices()
        self.input_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_combo.currentIndexChanged.connect(self._on_device_changed)

        # Detected Note, BPM, and 4-dot beat indicator (§56)
        detected_layout = QHBoxLayout()
        self.note_label = QLabel("--")
        note_font = QFont()
        note_font.setPointSize(28)
        note_font.setBold(True)
        self.note_label.setFont(note_font)

        self.bpm_label = QLabel("-- BPM")
        bpm_font = QFont()
        bpm_font.setPointSize(20)
        bpm_font.setBold(True)
        self.bpm_label.setFont(bpm_font)
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        detected_layout.addWidget(self.note_label)
        detected_layout.addStretch(1)
        detected_layout.addWidget(self.bpm_label)
        root_layout.addLayout(detected_layout)

        beat_row = QHBoxLayout()
        self.beat_dots: list[QLabel] = []
        dot_font = QFont()
        dot_font.setPointSize(14)
        dot_font.setBold(True)
        for beat_num in range(1, 5):
            dot = QLabel(f"o {beat_num}")
            dot.setFont(dot_font)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.beat_dots.append(dot)
            beat_row.addWidget(dot)
        root_layout.addLayout(beat_row)

        # Style selector
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style"))
        self.style_combo = QComboBox()
        for preset in STYLE_PRESETS.values():
            self.style_combo.addItem(preset.label, userData=preset.name)
        initial_style_idx = self.style_combo.findData(self._base_config.drums.style)
        if initial_style_idx >= 0:
            self.style_combo.setCurrentIndex(initial_style_idx)
        self.style_combo.currentIndexChanged.connect(self._on_style_changed)
        style_row.addWidget(self.style_combo, stretch=1)
        root_layout.addLayout(style_row)

        # Drums ON/OFF + volume & AI Music ON/OFF + volume (§56)
        mix_grid = QGridLayout()
        mix_grid.setHorizontalSpacing(10)
        mix_grid.setVerticalSpacing(8)

        self.drums_toggle = QPushButton()
        self.drums_toggle.setCheckable(True)
        self.drums_toggle.setChecked(not self._base_config.audio.drums_muted)
        self._sync_toggle_text(self.drums_toggle)
        self.drums_toggle.toggled.connect(self._on_drums_toggled)

        self.drums_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.drums_volume_slider.setRange(0, 2 * SLIDER_SCALE)
        self.drums_volume_slider.setValue(round(self._base_config.audio.drums_gain * SLIDER_SCALE))
        self.drums_volume_slider.valueChanged.connect(self._on_drums_volume_changed)

        self.ai_toggle = QPushButton()
        self.ai_toggle.setCheckable(True)
        self.ai_toggle.setChecked(not self._base_config.audio.ai_muted)
        self._sync_toggle_text(self.ai_toggle)
        self.ai_toggle.toggled.connect(self._on_ai_toggled)

        self.ai_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.ai_volume_slider.setRange(0, 2 * SLIDER_SCALE)
        self.ai_volume_slider.setValue(round(self._base_config.audio.ai_gain * SLIDER_SCALE))
        self.ai_volume_slider.valueChanged.connect(self._on_ai_volume_changed)

        mix_grid.addWidget(QLabel("Drums"), 0, 0)
        mix_grid.addWidget(self.drums_toggle, 0, 1)
        mix_grid.addWidget(self.drums_volume_slider, 0, 2)
        mix_grid.addWidget(QLabel("AI Music"), 1, 0)
        mix_grid.addWidget(self.ai_toggle, 1, 1)
        mix_grid.addWidget(self.ai_volume_slider, 1, 2)
        mix_grid.setColumnStretch(2, 1)
        root_layout.addLayout(mix_grid)

        # Follow <-> Improvise slider (§56, §74)
        fi_row = QHBoxLayout()
        fi_row.addWidget(QLabel("Follow"))
        self.follow_improvise_slider = QSlider(Qt.Orientation.Horizontal)
        self.follow_improvise_slider.setRange(0, SLIDER_SCALE)
        self.follow_improvise_slider.setValue(
            round(self._base_config.ai.follow_improvise * SLIDER_SCALE)
        )
        self.follow_improvise_slider.valueChanged.connect(self._on_follow_improvise_changed)
        fi_row.addWidget(self.follow_improvise_slider, stretch=1)
        fi_row.addWidget(QLabel("Improvise"))
        root_layout.addLayout(fi_row)

        # START / STOP JAM primary button + Record toggle + Debug panel toggle
        action_row = QHBoxLayout()
        self.jam_button = QPushButton("START JAM")
        jam_font = QFont()
        jam_font.setPointSize(14)
        jam_font.setBold(True)
        self.jam_button.setFont(jam_font)
        self.jam_button.setMinimumHeight(44)
        self.jam_button.clicked.connect(self.toggle_jam)

        self.record_toggle = QPushButton()
        self.record_toggle.setCheckable(True)
        self.record_toggle.setChecked(self._base_config.recording.enabled)
        self._sync_record_text()
        self.record_toggle.toggled.connect(self._on_record_toggled)

        self.debug_toggle = QPushButton("Debug")
        self.debug_toggle.setCheckable(True)
        self.debug_toggle.setChecked(False)
        self.debug_toggle.toggled.connect(self._on_debug_toggled)

        action_row.addWidget(self.jam_button, stretch=1)
        action_row.addWidget(self.record_toggle)
        action_row.addWidget(self.debug_toggle)
        root_layout.addLayout(action_row)

        # Developer diagnostics panel (§138, hidden by default per §139)
        self.debug_panel = DebugPanel(self)
        self.debug_panel.setVisible(False)
        root_layout.addWidget(self.debug_panel)

        # ~30 Hz poll timer on the GUI thread
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(UI_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self.poll_state)
        self._poll_timer.start()

        if preload_ai and self._warmup_fn is not None:
            self._start_background_warmup()
        self.poll_state()

    @property
    def session(self) -> JamSession | None:
        return self._session

    @property
    def is_jamming(self) -> bool:
        return self._session is not None and self._session.is_running

    @property
    def selected_style(self) -> StyleFamily:
        data = self.style_combo.currentData()
        return data if data in STYLE_PRESETS else "rock"

    @property
    def selected_input_device_name(self) -> str | None:
        data = self.input_combo.currentData()
        return str(data) if data is not None else None

    @property
    def selected_output_device_name(self) -> str | None:
        data = self.output_combo.currentData()
        return str(data) if data is not None else None

    @property
    def drums_gain(self) -> float:
        return self.drums_volume_slider.value() / SLIDER_SCALE

    @property
    def ai_gain(self) -> float:
        return self.ai_volume_slider.value() / SLIDER_SCALE

    @property
    def follow_improvise(self) -> float:
        return self.follow_improvise_slider.value() / SLIDER_SCALE

    def current_preferences(self) -> UserPreferences:
        """Build a `UserPreferences` snapshot from the current UI controls (§85)."""
        return UserPreferences(
            input_device_name=self.selected_input_device_name,
            output_device_name=self.selected_output_device_name,
            block_size_frames=self._base_config.audio.block_size_frames,
            style=self.selected_style,
            drums_gain=self.drums_gain,
            ai_gain=self.ai_gain,
        )

    def toggle_jam(self) -> None:
        """Start or stop the active jam session."""
        if self.is_jamming:
            self.stop_jam()
        else:
            self.start_jam()

    def start_jam(self) -> bool:
        """Build and start a new `JamSession` using current UI settings. Returns True on success."""
        if self._session is not None:
            self.stop_jam()

        self._last_error = None
        config = self._build_session_config()
        try:
            session = self._session_factory(config)
            session.set_style(self.selected_style)
            session.set_follow_improvise(self.follow_improvise)
            session.start()
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("Failed to start jam session: %s", exc)
            self._session = None
            self.poll_state()
            return False

        self._session = session
        self.jam_button.setText("STOP JAM")
        self.input_combo.setEnabled(False)
        self.output_combo.setEnabled(False)
        self.poll_state()
        return True

    def stop_jam(self) -> None:
        """Stop the current `JamSession` in §84 order and re-enable device selectors."""
        session = self._session
        self._session = None
        if session is not None:
            session.stop()
        self.jam_button.setText("START JAM")
        self.input_combo.setEnabled(True)
        self.output_combo.setEnabled(True)
        self.poll_state()

    def poll_state(self) -> None:
        """Read the latest `MusicalState` snapshot and update widgets on the GUI thread."""
        if (
            self._warming_up
            and self._warmup_thread is not None
            and not self._warmup_thread.is_alive()
        ):
            self._warming_up = False
            self.warmup_progress.setVisible(False)

        state = self._session.state if self._session is not None else MusicalState()
        self.note_label.setText(
            "--" if state.current_note is None else midi_to_name(state.current_note)
        )
        if state.bpm is not None:
            self.bpm_label.setText(f"{state.bpm:.0f} BPM")
        else:
            self.bpm_label.setText("-- BPM")

        for idx, dot in enumerate(self.beat_dots, start=1):
            is_active = state.beat == idx and self.is_jamming
            symbol = "O" if is_active else "o"
            dot.setText(f"{symbol} {idx}")

        self.status_label.setText(
            format_plain_status(
                session=self._session,
                warming_up=self._warming_up,
                warmup_failed=self._warmup_failed,
                last_error=self._last_error,
            )
        )
        self.debug_panel.update_diagnostics(DebugDiagnostics.from_session(self._session))

    def closeEvent(self, event: QCloseEvent) -> None:
        """Full §84 shutdown on window close: stop timer, stop session, save prefs, close MLX."""
        self._poll_timer.stop()
        self.stop_jam()
        self._save_preferences_if_enabled()
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            self._warmup_thread.join(timeout=5.0)
        self._warmup_thread = None
        if self._shutdown_mlx_on_close:
            shutdown_mlx_runtime()
        super().closeEvent(event)

    def _populate_devices(self) -> None:
        self.input_combo.clear()
        self.output_combo.clear()
        self.input_combo.addItem("System Default", userData=None)
        self.output_combo.addItem("System Default", userData=None)

        try:
            inventory = self._inventory_provider()
        except Exception as exc:
            logger.warning("Could not query audio devices for UI: %s", exc)
            return

        for dev in inventory.devices:
            if dev.is_input:
                self.input_combo.addItem(dev.name, userData=dev.name)
            if dev.is_output:
                self.output_combo.addItem(dev.name, userData=dev.name)

        self._select_combo_matching(self.input_combo, self._base_config.audio.input_device_name)
        self._select_combo_matching(self.output_combo, self._base_config.audio.output_device_name)

    @staticmethod
    def _select_combo_matching(combo: QComboBox, query: str | None) -> None:
        if not query:
            combo.setCurrentIndex(0)
            return
        needle = query.lower()
        for i in range(combo.count()):
            data = combo.itemData(i)
            if isinstance(data, str) and needle in data.lower():
                combo.setCurrentIndex(i)
                return
        # Keep literal query so custom/substring names from CLI still resolve.
        combo.addItem(query, userData=query)
        combo.setCurrentIndex(combo.count() - 1)

    def _build_session_config(self) -> JamMateConfig:
        style = self.selected_style
        audio_cfg = self._base_config.audio.model_copy(
            update={
                "input_device_name": self.selected_input_device_name,
                "output_device_name": self.selected_output_device_name,
                "drums_gain": self.drums_gain,
                "drums_muted": not self.drums_toggle.isChecked(),
                "ai_gain": self.ai_gain,
                "ai_muted": not self.ai_toggle.isChecked(),
            }
        )
        drum_cfg = self._base_config.drums.model_copy(update={"style": style})
        ai_cfg = self._base_config.ai.model_copy(
            update={
                "enabled": self._base_config.ai.enabled or self.ai_toggle.isChecked(),
                "style": style,
                "follow_improvise": self.follow_improvise,
            }
        )
        recording_cfg = self._base_config.recording.model_copy(
            update={"enabled": self.record_toggle.isChecked()}
        )
        return self._base_config.model_copy(
            update={
                "audio": audio_cfg,
                "drums": drum_cfg,
                "ai": ai_cfg,
                "recording": recording_cfg,
            }
        )

    def _start_background_warmup(self) -> None:
        if self._warmup_fn is None:
            return
        self._warming_up = True
        self.warmup_progress.setVisible(True)
        model_size = self._base_config.ai.model_size
        warmup_fn = self._warmup_fn

        def _worker() -> None:
            try:
                warmup_fn(model_size)
            except Exception as exc:
                self._warmup_failed = True
                logger.warning("Background AI model warmup failed: %s", exc)

        self._warmup_thread = threading.Thread(
            target=_worker,
            name="jammate-ai-warmup",
            daemon=False,
        )
        self._warmup_thread.start()

    @staticmethod
    def _sync_toggle_text(button: QPushButton) -> None:
        button.setText("ON" if button.isChecked() else "OFF")

    def _sync_record_text(self) -> None:
        self.record_toggle.setText("Recording" if self.record_toggle.isChecked() else "Record")

    def _save_preferences_if_enabled(self) -> None:
        if self._preferences_path is not None:
            save_preferences(self.current_preferences(), self._preferences_path)

    def _on_device_changed(self, _index: int) -> None:
        self._save_preferences_if_enabled()

    def _on_style_changed(self, _index: int) -> None:
        if self._session is not None:
            self._session.set_style(self.selected_style)
        self._save_preferences_if_enabled()

    def _on_drums_toggled(self, checked: bool) -> None:
        self._sync_toggle_text(self.drums_toggle)
        if self._session is not None:
            self._session.engine.processor.drums_muted = not checked

    def _on_drums_volume_changed(self, _value: int) -> None:
        if self._session is not None:
            self._session.engine.processor.drums_gain = self.drums_gain
        self._save_preferences_if_enabled()

    def _on_ai_toggled(self, checked: bool) -> None:
        self._sync_toggle_text(self.ai_toggle)
        if self._session is not None:
            self._session.engine.processor.ai_muted = not checked

    def _on_ai_volume_changed(self, _value: int) -> None:
        if self._session is not None:
            self._session.engine.processor.ai_gain = self.ai_gain
        self._save_preferences_if_enabled()

    def _on_follow_improvise_changed(self, _value: int) -> None:
        if self._session is not None:
            self._session.set_follow_improvise(self.follow_improvise)

    def _on_record_toggled(self, checked: bool) -> None:
        self._sync_record_text()
        if self._session is not None and self._session.is_running:
            if checked:
                self._session.start_recording()
            else:
                self._session.stop_recording()

    def _on_debug_toggled(self, checked: bool) -> None:
        self.debug_panel.setVisible(checked)
        self.adjustSize()


def run_gui(
    config: JamMateConfig,
    *,
    session_factory: SessionFactory = JamSession,
    inventory_provider: InventoryProvider = query_inventory,
    warmup_fn: ModelWarmupFn | None = default_mrt2_system_factory,
    preload_ai: bool = True,
    preferences_path: Path | None = DEFAULT_PREFERENCES_PATH,
) -> int:
    """Create (or reuse) `QApplication`, show `MainWindow`, and run the Qt event loop."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow(
        config,
        session_factory=session_factory,
        inventory_provider=inventory_provider,
        warmup_fn=warmup_fn,
        preload_ai=preload_ai,
        preferences_path=preferences_path,
    )
    window.show()
    if owns_app:
        return int(app.exec())
    return 0

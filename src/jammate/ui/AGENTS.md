# ui/ — PySide6 / Qt Widgets desktop UI

Provides the musician-facing `MainWindow` (§56, §139) and the toggleable developer `DebugPanel`
(§138).

| Module | Status | Thread |
|---|---|---|
| `main_window.py` | done (4.4, 4.5, 5.1) | GUI thread only (`QApplication` main thread) |
| `debug_panel.py` | done (4.4, 5.1) | GUI thread only |

## Invariants

- All Qt widget creation and updates happen exclusively on the GUI thread.
- The audio callback, analysis worker, drum scheduler, AI worker, and recorder writer **never**
  import or call Qt (§21, §22).
- A ~30 Hz `QTimer` on the GUI thread polls `JamSession.state` (frozen `MusicalState` snapshots) and
  non-blocking diagnostics (`engine.stats()`, `worker.stats()`, `drums`, `ai.stats()`).
- User controls update the running `JamSession` only through thread-safe setters (`set_instrument`,
  `set_style`, `set_follow_improvise`, `start_recording`, `stop_recording`, and atomic bus fields
  `drums_muted`, `drums_gain`, `ai_muted`, `ai_gain`).
- When `preferences_path` is provided, `MainWindow` loads `UserPreferences` (`settings.json`,
  including `instrument`) at startup and atomically saves them on instrument, device, style, or
  volume changes and on `closeEvent` (§85).
- Closing `MainWindow` (`closeEvent`) stops any active `JamSession` in §84 order, joins any
  background warmup thread, and shuts down the MLX runtime executor so zero background threads or
  open audio streams remain.

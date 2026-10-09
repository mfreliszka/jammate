# scripts/ — developer tools

Standalone tools for development and tuning. They are not part of the app. The app never imports
them, and they may do things the app avoids, such as blocking disk I/O on the main thread.

| Script | Step | Purpose |
|---|---|---|
| `capture_input.py` | 1.4 | Record the raw instrument input to WAV (fixtures for `tests/data/`) |
| `analyze_wav.py` | 2.2 (done) | Run the analysis pipeline over a WAV and print the event timeline (`--hops`: every estimate) |
| `make_placeholder_kit.py` | 3.2 | Generate license-safe placeholder drum samples |
| `render_jam.py` | 3.5 | Render a riff WAV through analysis and the drummer to a mixed WAV |

## Rules

- Run with `uv run python scripts/<name>.py …`. Each script has `--help` and a usage example in
  its docstring.
- Reuse the app's code (`jammate.*`) instead of duplicating logic. A script is a thin driver.
- Realtime rules still apply to anything that reaches the audio callback. Scripts drain rings on
  the main thread, never in the callback.
- Keep testable logic in plain functions. Tests load a script by path (see
  `tests/scripts/`), because `scripts/` is not a package.
- Never write into `tests/data/` without being asked. Keep fixtures short (under about 30 s),
  mono and 48 kHz.

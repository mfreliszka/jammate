# audio/ — audio transport

Owns everything that touches PortAudio: device discovery, the realtime stream and its callback,
the ring buffers between the callback and workers, and recording. It does no musical analysis
and makes no musical decisions.

| Module | Status | Thread |
|---|---|---|
| `devices.py` | done (1.2) | main/GUI thread only, before a stream opens |
| `ring_buffer.py` | done (1.3) | written by the callback, read by one worker |
| `engine.py` | done (1.3) | `AudioEngine` on the main thread; `BlockProcessor` on the PortAudio thread |
| `levels.py` | done (2.2) | any thread; pure dBFS math for meters and gates |
| `recorder.py` | Step 4.5 | callback → bounded ring → writer thread |

## Invariants

- Select devices by name or a unique part of it (`resolve_device`), never by a stored index.
  PortAudio indices change when devices are plugged in or out.
- PortAudio queries (`query_devices`, `check_*_settings`) are slow and not realtime-safe. Never
  call them from the callback.
- The root `AGENTS.md` realtime rules apply to everything the callback reaches.
- `RingBuffer` is strictly single-producer/single-consumer. Each ring has exactly one writer
  thread and one reader thread. Give a second consumer its own ring.
- `BlockProcessor.process` writes **all** of `outdata` through the monitor bus. Later buses
  (drums in Step 3.2, AI in Step 4.1) must add in place (`np.add(outdata, bus, out=outdata)`),
  never overwrite.
- Counters (`input_overflows`, `output_underflows`, `startup_xruns`, `errors`) are written only
  by the callback. Other threads only read them.

## PortAudio / CoreAudio notes

- PortAudio enumerates devices once, when it initializes. A device plugged in after start-up only
  appears after `sd._terminate()` + `sd._initialize()`, and that is only safe while no stream is
  open. The GUI device refresh (Step 4.4) needs this.
- To read one specific input channel without opening all of them, use
  `sd.CoreAudioSettings(channel_map=[...])` as the stream's `extra_settings`.
- `sd.CoreAudioSettings(fail_if_conversion_required=True)` did **not** stop the streams opening
  at 44.1 or 96 kHz on the developer's interface. Instead, `AudioEngine` checks
  `stream.samplerate` after opening and refuses a mismatch. Never change the sample rate silently
  (§141).
- `sd.CoreAudioSettings(change_device_parameters=True)` made no latency difference on that
  interface, so the engine doesn't use it. It can disturb other apps using the device.
- `check_input_settings` / `check_output_settings` do **not** validate sample rates on CoreAudio.
  They still catch channel and direction mismatches. With the developer's "USB Audio CODEC",
  44.1, 48 and 96 kHz all passed, even with `fail_if_conversion_required=True`. Real input
  streams also opened at all three rates. Only the reported latency showed the native rate:
  - 48 kHz: 16.4 ms;
  - 44.1 kHz: 29.5 ms;
  - 96 kHz: 48.2 ms.
  
  To verify the rate, open the stream, then log `stream.samplerate` and `stream.latency`
  (Step 1.3).
- One physical interface can appear as **two PortAudio devices with the same name**, one
  input-only and one output-only. `resolve_device` resolves per kind, so `--input usb
  --output usb` works. The duplex stream must pass both indices
  (`device=(input_index, output_index)`).
- Measured on "USB Audio CODEC" (duplex, 48 kHz):
  - PortAudio-reported latency is 16.4 ms in + 7.4 ms out at 128-frame blocks, and
    29.8 + 12.8 ms at 256;
  - the callback arrives every 2.67 ms with a p99 gap of 2.74 ms;
  - `BlockProcessor.process` costs about 9 µs mean and under 45 µs max per 128-frame block.
- Every stream start reports an input overflow on its first block or two. The engine counts
  xruns in the first `STARTUP_SETTLE_SECONDS` after each start as `startup_xruns`. Rare mid-run
  input overflows were also seen:
  - interface in + interface out: about 1 per 1–2 minutes;
  - interface in + MacBook speakers out: 0 in 170 s.
  
  The callback is not the cause. The suspect is PortAudio bridging the interface's two
  same-named devices. This is still open (see the journal's Step 1.4 carry-over).
- The input stream starts and ends with exact digital zeros: 4 blocks (512 frames) after
  `start()` and about 26 blocks (3,328 frames) while closing, on every capture so far. Analysis
  sees the same zeros in the live ring, so the step from zeros to the noise floor must not count
  as an onset.

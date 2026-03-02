#!/usr/bin/env python3
"""
ESP32 Serial Evaluation Driver (Sudoku).

Communicates with the ESP32 firmware via serial (UART) to run full inference
on the Sudoku test subset and collect predictions + timing data.

For Sudoku: seq_len is fixed at 81, vocab_size=11.
The firmware prepends puzzle_emb internally (16 positions), so internal
seq_len = 97 and inference takes ~4-6 minutes per puzzle.

Protocol:
  1. Send 'E' to enter eval mode
  2. Wait for "READY" response
  3. For each puzzle:
     - Send: [uint16_t seq_len] [uint8_t token_0] ... [uint8_t token_{seq_len-1}]
     - Receive: [uint16_t seq_len] [uint8_t pred_0] ... [uint8_t pred_{seq_len-1}] [uint32_t time_ms]
  4. Send seq_len=0 to exit eval mode

IMPORTANT: Close idf.py monitor BEFORE running this script. Only one
process can read from the serial port at a time.

Usage:
    python scripts/evaluate_serial.py \\
        --port /dev/ttyACM0 \\
        --test-subset test_subset.json \\
        --output results/results_esp32.json

Run from the esp32_trm repo root.
"""

import argparse
import json
import os
import struct
import sys
import time

import serial


def drain_serial(ser, wait_s=0.5):
    """Drain all pending bytes from serial buffer."""
    time.sleep(wait_s)
    discarded = 0
    while ser.in_waiting > 0:
        data = ser.read(ser.in_waiting)
        discarded += len(data)
        time.sleep(0.05)
    return discarded


def wait_for_ready(ser, timeout=60):
    """Wait for READY response from ESP32, then drain trailing bytes."""
    start = time.time()
    buf = b""
    while time.time() - start < timeout:
        data = ser.read(ser.in_waiting or 1)
        if data:
            buf += data
            if b"READY" in buf:
                # Drain any trailing bytes (e.g. \r\n after READY)
                discarded = drain_serial(ser, wait_s=0.3)
                return True
    return False


def wait_for_token(ser, token: bytes, timeout=60):
    """Wait until a specific token appears in the incoming stream."""
    start = time.time()
    buf = b""
    while time.time() - start < timeout:
        data = ser.read(ser.in_waiting or 1)
        if data:
            buf += data
            if token in buf:
                drain_serial(ser, wait_s=0.2)
                return True
        else:
            time.sleep(0.01)
    return False


class LineReader:
    """Buffered line reader over serial that preserves leftover bytes."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""

    def readline(self, deadline_s):
        """Return one line (ending with \\n) or None on timeout."""
        while time.time() < deadline_s:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                return line + b"\n"
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buf += chunk
            else:
                time.sleep(0.01)
        return None


def _hex_preview(b, max_len=32):
    b = b[:max_len]
    return " ".join(f"{x:02x}" for x in b)


def read_exact(ser, n, timeout):
    """Read exactly n bytes from serial, with timeout. Returns (bytes, complete)."""
    start = time.time()
    buf = b""
    while len(buf) < n:
        elapsed = time.time() - start
        if elapsed > timeout:
            return buf, False

        remaining = n - len(buf)
        data = ser.read(min(remaining, 4096))
        if data:
            buf += data
        else:
            time.sleep(0.01)

    return buf, True


def send_puzzle(ser, input_tokens, timeout=3600):
    """
    Send puzzle to ESP32 and receive prediction.

    Returns (pred_tokens, time_ms) or (None, None) on error.
    """
    seq_len = len(input_tokens)

    # Send seq_len (uint16_t LE) + input tokens
    payload = struct.pack("<H", seq_len) + bytes(input_tokens)
    ser.write(payload)
    ser.flush()

    # Receive response: [uint16_t seq_len] [pred_tokens] [uint32_t time_ms]
    expected_bytes = 2 + seq_len + 4

    buf, complete = read_exact(ser, expected_bytes, timeout)
    if not complete:
        return None, None, {
            "type": "timeout",
            "expected_bytes": expected_bytes,
            "received_bytes": len(buf),
            "tail_hex": _hex_preview(buf[-32:]),
        }

    # Parse response
    resp_len = struct.unpack("<H", buf[0:2])[0]
    if resp_len != seq_len:
        print(f"  WARNING: response seq_len mismatch: sent {seq_len}, got {resp_len}")
        print(f"           head_hex={_hex_preview(buf[:32])}")
        print(f"           tail_hex={_hex_preview(buf[-32:])}")
        # Try to recover: drain and return error
        drain_serial(ser, wait_s=2.0)
        return None, None, {
            "type": "len_mismatch",
            "sent_seq_len": seq_len,
            "resp_seq_len": resp_len,
            "head_hex": _hex_preview(buf[:64]),
            "tail_hex": _hex_preview(buf[-64:]),
        }

    pred_tokens = list(buf[2:2 + seq_len])
    time_ms = struct.unpack("<I", buf[2 + seq_len:2 + seq_len + 4])[0]

    return pred_tokens, time_ms, None


def send_puzzle_textproto(line_reader: LineReader, input_tokens, timeout=3600, act_steps=1):
    """
    Send puzzle using line-based text protocol (firmware 'T' mode).
    Sends: "<seq_len> [act_steps]\\n" followed by tokens.
    Expects response lines:
      TIME_MS <ms>
      STEPS_USED <n>
      PRED <p0> <p1> ... <p{seq_len-1}>
    Returns (pred_tokens, time_ms, err_info) or (pred_tokens, time_ms, err_info, steps_used).
    """
    ser = line_reader.ser
    seq_len = len(input_tokens)
    deadline = time.time() + timeout

    if act_steps > 1:
        ser.write(f"{seq_len} {act_steps}\n".encode("ascii"))
    else:
        ser.write(f"{seq_len}\n".encode("ascii"))
    ser.write((" ".join(str(x) for x in input_tokens) + "\n").encode("ascii"))
    ser.flush()

    line1 = line_reader.readline(deadline)
    if line1 is None:
        return None, None, {"type": "timeout", "stage": "time_ms_line"}, None
    line2 = line_reader.readline(deadline)
    if line2 is None:
        return None, None, {"type": "timeout", "stage": "steps_used_line"}, None
    line3 = line_reader.readline(deadline)
    if line3 is None:
        return None, None, {"type": "timeout", "stage": "pred_line"}, None

    try:
        s1 = line1.decode("utf-8", errors="replace").strip()
        s2 = line2.decode("utf-8", errors="replace").strip()
        s3 = line3.decode("utf-8", errors="replace").strip()

        if not s1.startswith("TIME_MS"):
            return None, None, {"type": "bad_reply", "line1": s1, "line2": s2, "line3": s3}, None

        time_ms = int(s1.split()[1])
        steps_used = 1

        if s2.startswith("STEPS_USED"):
            steps_used = int(s2.split()[1])
            pred_line = s3
        elif s2.startswith("PRED"):
            pred_line = s2
        else:
            return None, None, {"type": "bad_reply", "line1": s1, "line2": s2, "line3": s3}, None

        if not pred_line.startswith("PRED"):
            return None, None, {"type": "bad_reply", "pred_line": pred_line}, None

        pred_strs = pred_line.split()[1:]
        if len(pred_strs) != seq_len:
            return None, None, {"type": "pred_len_mismatch", "expected": seq_len, "got": len(pred_strs), "line": pred_line[:200]}, None
        pred_tokens = [int(x) & 0xFF for x in pred_strs]
        return pred_tokens, time_ms, None, steps_used
    except Exception as e:
        return None, None, {"type": "parse_error", "error": repr(e)}, None


def auto_load_and_enter_eval(ser, *, load_timeout_s: int = 180):
    """
    Handle the case where the ESP32 resets on serial connect.
    Sends 'l' to load model, waits, then sends 'E' for eval mode.
    """
    print("Attempting to auto-load model and enter eval mode...")

    # Wait for the ESP32 to boot and show menu
    time.sleep(3)
    drain_serial(ser, wait_s=0.5)

    # Send 'l' to load model
    print("  Sending 'l' to load model...")
    ser.write(b"l")
    ser.flush()

    # Wait for model to load (can take a few seconds)
    start = time.time()
    buf = b""
    while time.time() - start < int(load_timeout_s):
        data = ser.read(ser.in_waiting or 1)
        if data:
            buf += data
            if b"Model loaded" in buf or b"already loaded" in buf:
                print("  Model loaded successfully.")
                break
            if b"Failed to load" in buf:
                print("  ERROR: Model failed to load on ESP32.")
                return False
    else:
        print(f"  WARNING: Timed out waiting for model load (got: {buf[-200:]})")
        return False

    drain_serial(ser, wait_s=0.5)

    # Send 'E' to enter eval mode
    print("  Sending 'E' for eval mode...")
    ser.write(b"E")
    ser.flush()

    if wait_for_ready(ser, timeout=30):
        return True
    else:
        print("  ERROR: Did not receive READY after sending 'E'.")
        return False


def main():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="ESP32 serial evaluation driver (Sudoku)")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                        help="Serial port")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate")
    parser.add_argument("--test-subset", type=str,
                        default=os.path.join(_root, "test_subset.json"),
                        help="Path to test subset JSON")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_root, "results", "results_esp32.json"),
                        help="Output results JSON")
    parser.add_argument("--resume-output", type=str, default="",
                        help="Resume from an existing results JSON: skip completed example_index entries and append new results")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Per-puzzle timeout in seconds")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of puzzles (0 = all)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from puzzle index (for resuming)")
    parser.add_argument("--auto-load", action="store_true",
                        help="Auto-load model on ESP32 (if ESP32 resets on connect)")
    parser.add_argument(
        "--auto-load-timeout",
        type=int,
        default=180,
        help="Timeout (seconds) to wait for firmware model load after sending 'l' (only with --auto-load).",
    )
    parser.add_argument("--protocol", type=str, default="binary", choices=["binary", "text"],
                        help="Evaluation protocol: binary uses 'E', text uses firmware 'T' line protocol (more robust)")
    parser.add_argument("--auto-unpack", action="store_true",
                        help="After auto-load, run 'u' to pre-unpack weights (faster inference, uses more PSRAM)")
    parser.add_argument("--continue-on-timeout", action="store_true",
                        help="Keep going after a timeout (not recommended; device will still be computing)")
    parser.add_argument("--act-steps", type=int, default=1,
                        help="Number of ACT steps (1=single pass, 16=haltable max). Only used with --protocol text")
    args = parser.parse_args()

    # Load test subset
    print(f"Loading test subset: {args.test_subset}")
    with open(args.test_subset) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    print(f"  {len(puzzles)} puzzles loaded")

    # Optional resume: load existing output and skip puzzles already completed.
    results = []
    total_time_ms = 0
    errors = 0
    done_example_indices = set()
    if args.resume_output and os.path.exists(args.resume_output):
        try:
            prev = json.load(open(args.resume_output, "r"))
            results = prev.get("results", []) or []
            errors = int(prev.get("errors", 0) or 0)
            total_time_ms = int(prev.get("total_esp_time_ms", 0) or 0)
            for r in results:
                if r.get("predictions") is not None and r.get("example_index") is not None:
                    done_example_indices.add(r["example_index"])
            if done_example_indices:
                puzzles = [p for p in puzzles if p.get("example_index") not in done_example_indices]
                print(f"  Resume: skipping {len(done_example_indices)} completed puzzles from {args.resume_output}")
        except Exception as e:
            print(f"  WARNING: failed to load resume output '{args.resume_output}': {e!r}")
            results = []
            total_time_ms = 0
            errors = 0
            done_example_indices = set()

    if args.limit > 0:
        puzzles = puzzles[:args.limit]
        print(f"  Limited to {len(puzzles)} puzzles")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Connect to ESP32
    print(f"\nConnecting to {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    # Some ESP32-S3 dev boards use DTR/RTS for auto-reset/boot strapping.
    # Leaving these asserted can accidentally hold the chip in download mode.
    try:
        ser.dtr = False
        ser.rts = False
    except Exception:
        pass
    time.sleep(2)  # Wait for connection to stabilize
    line_reader = LineReader(ser)

    # Drain any pending data from boot/reset
    discarded = drain_serial(ser, wait_s=1.0)
    if discarded:
        print(f"  Drained {discarded} bytes from boot output")

    if args.auto_load:
        if args.protocol == "binary":
            ok = auto_load_and_enter_eval(ser, load_timeout_s=int(args.auto_load_timeout))
        else:
            # Auto-load, then enter text protocol mode 'T'
            print("Attempting to auto-load model and enter text eval mode...")
            time.sleep(3)
            drain_serial(ser, wait_s=0.5)
            print("  Sending 'l' to load model...")
            ser.write(b"l")
            ser.flush()
            start = time.time()
            buf = b""
            ok = False
            while time.time() - start < int(args.auto_load_timeout):
                data = ser.read(ser.in_waiting or 1)
                if data:
                    buf += data
                    if b"Model loaded" in buf or b"already loaded" in buf:
                        print("  Model loaded successfully.")
                        ok = True
                        break
                    if b"Failed to load" in buf:
                        print("  ERROR: Model failed to load on ESP32.")
                        ok = False
                        break
                else:
                    time.sleep(0.01)
            if not ok:
                print(f"  ERROR: Timed out waiting for model load (tail={buf[-200:]!r})")
            else:
                drain_serial(ser, wait_s=0.5)
                print("  Sending 'T' for text eval mode...")
                ser.write(b"T")
                ser.flush()
                ok = wait_for_token(ser, b"READYTXT", timeout=30)
                if not ok:
                    print("  ERROR: Did not receive READYTXT after sending 'T'.")
        if not ok:
            ser.close()
            sys.exit(1)
        if args.auto_unpack:
            # We are currently in eval mode; exit, unpack, then re-enter eval mode.
            # NOTE: Exit framing differs between binary vs text protocols.
            print("Auto-unpack requested; exiting eval mode to run 'u'...")
            if args.protocol == "binary":
                ser.write(struct.pack("<H", 0))
            else:
                ser.write(b"0\n")
            ser.flush()
            time.sleep(1)
            drain_serial(ser, wait_s=0.5)

            print("Sending 'u' to pre-unpack weights...")
            ser.write(b"u")
            ser.flush()

            # Wait for completion message (or failure)
            start = time.time()
            buf = b""
            ok = False
            while time.time() - start < 120:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    buf += data
                    if b"successfully" in buf or b"already" in buf:
                        ok = True
                        break
                    if b"Failed" in buf:
                        break
                else:
                    time.sleep(0.05)
            if ok:
                print("  Weights pre-unpacked.")
            else:
                print(f"  WARNING: pre-unpack may have failed or timed out (tail={buf[-200:]!r})")
            drain_serial(ser, wait_s=0.5)

            if args.protocol == "binary":
                print("Re-entering eval mode (sending 'E')...")
                ser.write(b"E")
                ser.flush()
                if not wait_for_ready(ser, timeout=30):
                    print("ERROR: Did not receive READY after re-entering eval mode.")
                    ser.close()
                    sys.exit(1)
            else:
                print("Re-entering text eval mode (sending 'T')...")
                ser.write(b"T")
                ser.flush()
                if not wait_for_token(ser, b"READYTXT", timeout=30):
                    print("ERROR: Did not receive READYTXT after re-entering text eval mode.")
                    ser.close()
                    sys.exit(1)
    else:
        # Try to enter eval mode directly (model should already be loaded)
        if args.protocol == "binary":
            print("Entering evaluation mode (sending 'E')...")
            ser.write(b"E")
            ser.flush()

            if not wait_for_ready(ser, timeout=30):
                print("ERROR: Did not receive READY from ESP32.")
                print("  Make sure the model is loaded (press 'l' in monitor first).")
                print("  Make sure idf.py monitor is CLOSED before running this script.")
                print("  Or use --auto-load to auto-load the model.")
                ser.close()
                sys.exit(1)
        else:
            print("Entering text evaluation mode (sending 'T')...")
            ser.write(b"T")
            ser.flush()
            if not wait_for_token(ser, b"READYTXT", timeout=30):
                print("ERROR: Did not receive READYTXT from ESP32.")
                ser.close()
                sys.exit(1)

    print("ESP32 is READY.")
    print(f"  Per-puzzle timeout: {args.timeout}s")
    if args.act_steps > 1:
        print(f"  ACT steps: {args.act_steps} (haltable)")
    print("  NOTE: if a puzzle times out, aborting is safest (ESP32 is still computing).\n")

    total_steps_used = 0

    # Evaluate puzzles (may be resuming with existing results)
    start_time = time.time()

    try:
        for i, puzzle in enumerate(puzzles):
            if i < args.start_from:
                continue

            seq_len = int(puzzle["seq_len"])
            input_tokens = puzzle["input_tokens"]

            t0 = time.time()
            steps_used = 1
            if args.protocol == "binary":
                pred_tokens, time_ms, err_info = send_puzzle(ser, input_tokens, timeout=args.timeout)
            else:
                pred_tokens, time_ms, err_info, steps_used = send_puzzle_textproto(
                    line_reader, input_tokens, timeout=args.timeout, act_steps=args.act_steps)
                if steps_used is None:
                    steps_used = 1
            t1 = time.time()

            if pred_tokens is None:
                wall = t1 - t0
                print(f"  [{i+1}/{len(puzzles)}] ERROR: timeout or protocol error "
                      f"(wall={wall:.0f}s)")
                errors += 1
                results.append({
                    "example_index": puzzle.get("example_index"),
                    "group_id": puzzle.get("group_id", 0),
                    "puzzle_identifier": puzzle.get("puzzle_identifier"),
                    "seq_len": seq_len,
                    "predictions": None,
                    "time_ms": None,
                    "act_steps": args.act_steps,
                    "steps_used": None,
                    "error": err_info["type"] if err_info else "error",
                    "error_info": err_info,
                })

                if not args.continue_on_timeout:
                    print("  Aborting run after timeout to avoid desync.")
                    break

                print("  Attempting to resync serial connection (best-effort)...")
                drain_serial(ser, wait_s=5.0)
                continue

            total_time_ms += time_ms
            total_steps_used += steps_used

            results.append({
                "example_index": puzzle.get("example_index"),
                "group_id": puzzle.get("group_id", 0),
                "puzzle_identifier": puzzle.get("puzzle_identifier"),
                "seq_len": seq_len,
                "predictions": pred_tokens,
                "time_ms": time_ms,
                "act_steps": args.act_steps,
                "steps_used": steps_used,
            })

            # Progress reporting
            completed = len([r for r in results if r.get("predictions") is not None])
            if completed > 0:
                avg_esp_time = total_time_ms / completed / 1000
                remaining_puzzles = len(puzzles) - i - 1
                eta_min = remaining_puzzles * avg_esp_time / 60
                avg_steps = total_steps_used / completed

                steps_info = f", steps={steps_used}" if args.act_steps > 1 else ""
                print(f"  [{i+1}/{len(puzzles)}] seq_len={seq_len:3d}, "
                      f"esp_time={time_ms/1000:.1f}s{steps_info}, "
                      f"wall={t1-t0:.1f}s, "
                      f"ETA={eta_min:.1f}min ({remaining_puzzles} remaining)")

            _save_results(args.output, results, errors, total_time_ms, start_time, args.port,
                          act_steps=args.act_steps, total_steps_used=total_steps_used)
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
        print("  NOTE: the ESP32 may still be computing the last puzzle and may later print its binary response.")
        print("  Safest next step is to reset the ESP32 before starting a new eval run.")

    # Exit eval mode
    print("\nExiting eval mode...")
    try:
        if args.protocol == "binary":
            ser.write(struct.pack("<H", 0))  # seq_len=0 = exit
            ser.flush()
        else:
            ser.write(b"0\n")
            ser.flush()
    except Exception:
        pass
    time.sleep(1)
    ser.close()

    # Final save
    _save_results(args.output, results, errors, total_time_ms, start_time, args.port,
                  act_steps=args.act_steps, total_steps_used=total_steps_used)

    total_elapsed = time.time() - start_time
    print(f"\nResults saved to: {args.output}")
    print(f"  Total puzzles:  {len(results)}")
    print(f"  Errors:         {errors}")
    if args.act_steps > 1:
        successful = len(results) - errors
        if successful > 0:
            print(f"  Avg steps used: {total_steps_used/successful:.2f}")
    print(f"  Total ESP time: {total_time_ms/1000:.1f}s ({total_time_ms/1000/60:.1f}min)")
    print(f"  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    successful = len(results) - errors
    if successful > 0:
        print(f"  Avg ESP time:   {total_time_ms/successful/1000:.1f}s per puzzle")


def _save_results(output_path, results, errors, total_time_ms, start_time, port,
                   act_steps=1, total_steps_used=0):
    """Save results incrementally."""
    total_elapsed = time.time() - start_time
    successful = len([r for r in results if r.get("predictions") is not None])
    output_data = {
        "config": "esp32_deployed",
        "port": port,
        "act_steps": act_steps,
        "haltable": act_steps > 1,
        "avg_steps_used": round(total_steps_used / successful, 2) if successful > 0 else None,
        "num_puzzles": len(results),
        "errors": errors,
        "total_esp_time_ms": total_time_ms,
        "total_wall_time_s": round(total_elapsed, 1),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hardware-free regression checks for the keyboard reader's lifetime and input."""
import os
from pathlib import Path
import pty
import select
import shutil
import subprocess
import tempfile
import termios
import time
import unittest


class KeyboardLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="g1-keyboard-test-")
        root = Path(cls.directory.name)
        cls.binary = root / "keyboard_test"
        source = root / "keyboard_test.cpp"
        source.write_text(r'''
#include "isaaclab/devices/keyboard/keyboard.h"
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "idle";
    Keyboard keyboard;
    if (mode == "prompt" && keyboard.getString("PROMPT\n") != "hello") return 2;
    std::cout << "READY" << std::endl;
    bool saw_up = false;
    const auto end = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(mode == "partial" ? 30 : 350);
    while (std::chrono::steady_clock::now() < end) {
        keyboard.update();
        saw_up |= keyboard.key() == "up";
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return (mode == "arrow" || mode == "prompt") && !saw_up ? 3 : 0;
}
''')
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler is required")
        include = Path(__file__).resolve().parents[1] / "deploy/include"
        subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(include),
                        str(source), "-o", str(cls.binary)], check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_closed_stdin_exit(self):
        result = subprocess.run([str(self.binary)], stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_terminal(self, mode="idle", data=b""):
        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        process = subprocess.Popen([str(self.binary), mode], stdin=slave,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def wait_for(marker):
            output = b""
            deadline = time.monotonic() + 3
            while marker not in output:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, output)
                readable, _, _ = select.select([process.stdout], [], [], remaining)
                self.assertTrue(readable, output)
                chunk = os.read(process.stdout.fileno(), 512)
                self.assertTrue(chunk, output)
                output += chunk
        try:
            if mode == "prompt":
                wait_for(b"PROMPT\n")
                os.write(master, b"hello\n")
            wait_for(b"READY\n")
            if data:
                os.write(master, data)
            _, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, stderr)
            restored = termios.tcgetattr(slave)
            # macOS may set PENDIN when restoring canonical mode; it is a
            # transient kernel input-reprocessing flag, not an input setting.
            restored[3] &= ~getattr(termios, "PENDIN", 0)
            original[3] &= ~getattr(termios, "PENDIN", 0)
            self.assertEqual(restored, original)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            os.close(master)
            os.close(slave)

    def test_idle_terminal_exit(self):
        self.run_terminal()

    def test_incomplete_escape_exit(self):
        self.run_terminal(mode="partial", data=b"\x1b")

    def test_arrow_key(self):
        self.run_terminal(mode="arrow", data=b"\x1b[A")

    def test_prompt_resumes_reader(self):
        self.run_terminal(mode="prompt", data=b"\x1b[A")


if __name__ == "__main__":
    unittest.main()

"""Camera password/jump tests. No SSH connections or real credentials are used."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import unittest

from stem_camera_ssh import PASSWORD_ENV, camera_ssh_command, prompt_password
from stem_instrument_readonly import HardwareReader, execute


def camera():
    return {"host": "192.168.111.90", "user": "root", "password": "camera-test-only!$'",
            "known_hosts_file": "/run/camera ssh/known_hosts",
            "jump_host": {"host": "mothership2", "user": "operator", "port": 2222,
                          "password": "jump-test-only!$'"}}


class CameraSshTest(unittest.TestCase):
    def test_distinct_passwords_and_no_password_in_command_arguments(self):
        settings = camera()
        argv, environment = camera_ssh_command(settings, 20)
        self.assertEqual(argv[-1], "root@192.168.111.90")
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("BatchMode=no", argv)
        self.assertEqual(environment["SSH_ASKPASS_REQUIRE"], "force")
        proxy = next(item.split("=", 1)[1] for item in argv if item.startswith("ProxyCommand="))
        jump = shlex.split(proxy)
        self.assertEqual(jump[-3:], ["-W", "[%h]:%p", "operator@mothership2"])
        self.assertIn("UserKnownHostsFile=/run/camera ssh/known_hosts", jump)
        self.assertEqual(jump[jump.index("-p") + 1], "2222")
        for value in (settings["password"], settings["jump_host"]["password"]):
            self.assertNotIn(value, " ".join(argv))
        self.assertEqual(prompt_password("root@192.168.111.90's password: ", environment), settings["password"])
        self.assertEqual(prompt_password("operator@mothership2's password: ", environment),
                         settings["jump_host"]["password"])

    def test_direct_password_and_both_mixed_authentication_directions(self):
        settings = camera()
        settings.pop("jump_host")
        argv, environment = camera_ssh_command(settings, 8)
        self.assertFalse(any(item.startswith("ProxyCommand=") for item in argv))
        self.assertEqual(len(json.loads(environment[PASSWORD_ENV])), 1)
        settings = camera()
        settings["jump_host"].pop("password")
        settings["jump_host"]["identity_file"] = "/keys/jump key"
        argv, environment = camera_ssh_command(settings, 8)
        proxy = shlex.split(next(item.split("=", 1)[1] for item in argv if item.startswith("ProxyCommand=")))
        self.assertIn("BatchMode=yes", proxy)
        self.assertIn("/keys/jump key", proxy)
        self.assertEqual(len(json.loads(environment[PASSWORD_ENV])), 1)
        settings = camera()
        settings.pop("password")
        settings["identity_file"] = "/keys/camera"
        argv, environment = camera_ssh_command(settings, 8)
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("/keys/camera", argv)
        self.assertEqual(prompt_password("operator@mothership2's password:", environment),
                         settings["jump_host"]["password"])

    def test_key_only_retains_noninteractive_behavior(self):
        argv, environment = camera_ssh_command({"host": "camera", "identity_file": "/key"}, 8)
        self.assertIsNone(environment)
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)

    def test_legacy_host_key_exception_is_opt_in_and_per_hop(self):
        settings = camera()
        argv, _ = camera_ssh_command(settings, 8)
        self.assertNotIn("HostKeyAlgorithms=", " ".join(argv))
        settings["host_key_algorithms"] = "+ssh-rsa"
        argv, _ = camera_ssh_command(settings, 8)
        self.assertIn("HostKeyAlgorithms=+ssh-rsa", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        proxy = shlex.split(next(item.split("=", 1)[1] for item in argv if item.startswith("ProxyCommand=")))
        self.assertNotIn("HostKeyAlgorithms=+ssh-rsa", proxy)
        self.assertIn("StrictHostKeyChecking=yes", proxy)
        settings.pop("host_key_algorithms")
        settings["jump_host"]["host_key_algorithms"] = "+ssh-rsa"
        argv, _ = camera_ssh_command(settings, 8)
        self.assertNotIn("HostKeyAlgorithms=+ssh-rsa", argv)
        proxy = shlex.split(next(item.split("=", 1)[1] for item in argv if item.startswith("ProxyCommand=")))
        self.assertIn("HostKeyAlgorithms=+ssh-rsa", proxy)

    def test_rejects_invalid_host_key_algorithm_options(self):
        for value in ("", True, ["ssh-rsa"], "+ssh-rsa -oStrictHostKeyChecking=no",
                      "+ssh-rsa\nProxyCommand=bad", "+ssh-rsa;bad"):
            settings = camera()
            settings["host_key_algorithms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera_ssh_command(settings, 8)

    def test_prompt_helper_never_answers_other_prompts(self):
        _, environment = camera_ssh_command(camera(), 8)
        for prompt in ("Password:", "root@other's password:", "Are you sure you want to continue connecting?",
                       "Enter passphrase for key '/key':", "Verification code:"):
            with self.subTest(prompt=prompt):
                self.assertIsNone(prompt_password(prompt, environment))
        self.assertIsNone(prompt_password("root@camera's password:", {PASSWORD_ENV: "broken json"}))

    def test_helper_process_and_password_quoting(self):
        settings = camera()
        _, environment = camera_ssh_command(settings, 8)
        helper = Path(environment["SSH_ASKPASS"])
        result = subprocess.run([sys.executable, str(helper), "root@192.168.111.90's password: "],
                                env=dict(os.environ, **environment), capture_output=True, check=True)
        self.assertEqual(result.stdout, (settings["password"] + "\n").encode())
        result = subprocess.run([sys.executable, str(helper), "unknown prompt"],
                                env=dict(os.environ, **environment), capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")

    def test_fixed_camera_input_and_error_redaction(self):
        settings = camera()
        calls = []

        def runner(argv, timeout, stdin=None, env=None):
            calls.append((argv, stdin, env))
            raise RuntimeError("failed " + settings["password"] + " / " + settings["jump_host"]["password"])

        reader = HardwareReader({"camera": settings}, runner)
        with self.assertRaises(RuntimeError) as caught:
            reader.camera_read("sensor temp")
        self.assertEqual(calls[0][1], "dsh sensor temp\n")
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertNotIn(settings["password"], str(caught.exception))
        self.assertNotIn(settings["jump_host"]["password"], str(caught.exception))
        self.assertNotIn(PASSWORD_ENV, os.environ)

    def test_executor_passes_environment_without_overwriting_parent(self):
        name = "STEM_SSH_ENV_TEST"
        self.assertEqual(execute([sys.executable, "-c", "import os; print(os.getenv('" + name + "'))"],
                                 2, env={name: "child-only"}).strip(), "child-only")
        self.assertNotIn(name, os.environ)

    @unittest.skipUnless(os.name == "posix", "POSIX SSH process groups")
    def test_timeout_also_terminates_proxy_children(self):
        command = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']); time.sleep(5)"
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            execute([sys.executable, "-c", command], .1)
        self.assertLess(time.monotonic() - started, 1)

    def test_rejects_invalid_or_unquoted_credentials_and_targets(self):
        for key, value in (("host", "camera;touch bad"), ("user", "root -oX"),
                           ("password", 1234), ("password", "two\nlines"),
                           ("port", True), ("jump_host", "operator@mothership2")):
            settings = camera()
            settings[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                camera_ssh_command(settings, 8)


if __name__ == "__main__":
    unittest.main()

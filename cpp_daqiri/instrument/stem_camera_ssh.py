#!/usr/bin/env python3
"""Build camera SSH connections and answer fixed OpenSSH password prompts."""

import json
import os
from pathlib import Path
import re
import shlex
import sys


PASSWORD_ENV = "STEM_CAMERA_SSH_PASSWORDS"


def destination_key(destination):
    user, host = destination.rsplit("@", 1)
    return user + "@" + host.lower()


def ssh_arguments(settings, timeout, default_user, default_known_hosts=None):
    if not isinstance(settings, dict):
        raise ValueError("camera SSH settings and jump_host must be mappings")
    host = settings.get("host", "")
    user = settings.get("user", default_user)
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", host):
        raise ValueError("invalid camera/jump SSH host")
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user):
        raise ValueError("invalid camera/jump SSH user")
    port = settings.get("port", 22)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("camera/jump SSH port must be an integer in [1,65535]")
    password = settings.get("password")
    if password is not None and (
        not isinstance(password, str) or not password or any(c in password for c in "\r\n\0")
    ):
        raise ValueError("camera/jump password must be a nonempty quoted, single-line YAML string")
    argv = ["ssh", "-T", "-o", "BatchMode=" + ("no" if password is not None else "yes"),
            "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no",
            "-o", "ConnectTimeout={}".format(max(1, int(timeout))), "-p", str(port)]
    host_key_algorithms = settings.get("host_key_algorithms")
    if host_key_algorithms is not None:
        if not isinstance(host_key_algorithms, str) or not re.fullmatch(
            r"[+^\-]?[A-Za-z0-9@._*\-]+(?:,[A-Za-z0-9@._*\-]+)*", host_key_algorithms
        ):
            raise ValueError("host_key_algorithms must be a nonempty SSH algorithm list without whitespace")
        argv += ["-o", "HostKeyAlgorithms=" + host_key_algorithms]
    if password is not None:
        argv += ["-o", "PreferredAuthentications=password", "-o", "PasswordAuthentication=yes",
                 "-o", "KbdInteractiveAuthentication=no", "-o", "PubkeyAuthentication=no",
                 "-o", "NumberOfPasswordPrompts=1"]
    elif settings.get("identity_file"):
        argv += ["-i", settings["identity_file"], "-o", "IdentitiesOnly=yes"]
    known_hosts = settings.get("known_hosts_file", default_known_hosts)
    if known_hosts:
        argv += ["-o", "UserKnownHostsFile=" + known_hosts]
    for argument in argv:
        if not isinstance(argument, str) or any(c in argument for c in "\r\n\0"):
            raise ValueError("invalid camera/jump SSH option")
    return argv, user + "@" + host, password


def camera_ssh_command(camera, timeout):
    argv, destination, password = ssh_arguments(camera, timeout, "root")
    passwords = {}
    if password is not None:
        passwords[destination_key(destination)] = password
    jump = camera.get("jump_host")
    if jump is not None:
        jump_argv, jump_destination, jump_password = ssh_arguments(
            jump, timeout, "root", camera.get("known_hosts_file")
        )
        if jump_password is not None:
            identifier = destination_key(jump_destination)
            if identifier in passwords and passwords[identifier] != jump_password:
                raise ValueError("camera and jump host cannot use different passwords for the same user@host")
            passwords[identifier] = jump_password
        # Explicit per-hop options are not inherited by ProxyJump. This is the
        # equivalent stdio forwarding connection, with each hop's own identity.
        jump_argv += ["-W", "[%h]:%p", jump_destination]
        proxy = shlex.join(jump_argv).replace("%", "%%").replace("[%%h]:%%p", "[%h]:%p")
        argv += ["-o", "ProxyCommand=" + proxy]
    argv.append(destination)
    if not passwords:
        return argv, None
    environment = {
        "SSH_ASKPASS": str(Path(__file__).resolve()),
        "SSH_ASKPASS_REQUIRE": "force", "LC_ALL": "C",
        PASSWORD_ENV: json.dumps(passwords),
    }
    return argv, environment


def prompt_password(prompt, environment):
    # Never answer host-key, private-key passphrase, MFA, or unknown-host prompts.
    match = re.fullmatch(r"([^\s@]+@[^\s]+)'s password:\s*", prompt)
    if not match:
        return None
    try:
        passwords = json.loads(environment.get(PASSWORD_ENV, "{}"))
        value = passwords.get(destination_key(match.group(1)))
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value and not any(c in value for c in "\r\n\0") else None


if __name__ == "__main__":
    value = prompt_password(sys.argv[1], os.environ) if len(sys.argv) == 2 else None
    if value is None:
        raise SystemExit(1)
    sys.stdout.buffer.write(value.encode("utf-8") + b"\n")

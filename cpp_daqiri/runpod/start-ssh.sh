#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != 0 ]]; then
    echo "Runpod SSH startup requires root to configure authorized keys." >&2
    exit 1
fi

if [[ -z "${PUBLIC_KEY:-}" ]]; then
    echo "Runpod did not provide PUBLIC_KEY; refusing to start SSH without a key." >&2
    exit 1
fi

install -d -m 700 /root/.ssh
printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
install -d -m 755 /run/sshd

echo "Runpod SSH ready on port 22; benchmark binary: /opt/stem_daqiri/bin/stem_daqiri_rx"
exec /usr/sbin/sshd -D -e \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o PermitRootLogin=prohibit-password \
    -o PubkeyAuthentication=yes

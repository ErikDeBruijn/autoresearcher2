#!/bin/bash
# Deploy autoresearcher2 services to VM.
# Run from laptop: ssh ... < deploy/setup-vm.sh
# Or on the VM directly: bash deploy/setup-vm.sh

set -euo pipefail

REPO=/root/github.com/erikdebruijn/autoresearcher2

echo "=== Updating repo ==="
cd "$REPO"
git fetch origin
git checkout web-ui-v4.1
git pull origin web-ui-v4.1

echo "=== Installing systemd services ==="
cp deploy/autoresearcher.service /etc/systemd/system/
cp deploy/autoresearcher-web.service /etc/systemd/system/
systemctl daemon-reload

echo "=== Stopping screen sessions (if any) ==="
screen -S v4research -X quit 2>/dev/null || true
screen -S webapi -X quit 2>/dev/null || true

echo "=== Starting services ==="
systemctl enable autoresearcher autoresearcher-web
systemctl restart autoresearcher-web
systemctl restart autoresearcher

echo "=== Status ==="
systemctl status autoresearcher --no-pager -l || true
systemctl status autoresearcher-web --no-pager -l || true

echo "=== Done ==="
echo "Logs: journalctl -u autoresearcher -f"
echo "Web:  journalctl -u autoresearcher-web -f"

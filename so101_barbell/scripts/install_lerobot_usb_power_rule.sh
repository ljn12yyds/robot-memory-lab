#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULE_SOURCE="${PROJECT_ROOT}/config/99-lerobot-usb-power.rules"
RULE_TARGET="/etc/udev/rules.d/99-lerobot-usb-power.rules"

sudo install -m 0644 "$RULE_SOURCE" "$RULE_TARGET"
sudo udevadm control --reload-rules

# Apply access immediately to controllers that are already connected. Future
# reconnects and ttyACM renumbering are handled automatically by the udev rule.
for tty_sysfs in /sys/class/tty/ttyACM*; do
  [[ -e "$tty_sysfs" ]] || continue
  tty_properties="$(udevadm info --query=property --path="$tty_sysfs")"
  if grep -qx 'ID_VENDOR_ID=1a86' <<<"$tty_properties" && \
     grep -qx 'ID_MODEL_ID=55d3' <<<"$tty_properties"; then
    tty_name="${tty_sysfs##*/}"
    sudo chgrp dialout "/dev/$tty_name"
    sudo chmod 0660 "/dev/$tty_name"
    echo "$tty_name serial_access=dialout:0660"
  fi
done

# Apply immediately to the currently enumerated robot hubs/controllers/cameras.
for device in /sys/bus/usb/devices/*; do
  [[ -f "${device}/idVendor" && -f "${device}/idProduct" ]] || continue
  vendor="$(<"${device}/idVendor")"
  product="$(<"${device}/idProduct")"
  case "${vendor}:${product}" in
    1a40:0101|1a86:55d3|1bcf:2281|1e45:0209)
      if [[ -e "${device}/power/control" ]]; then
        printf 'on\n' | sudo tee "${device}/power/control" >/dev/null
        printf '%s %s:%s power=%s\n' \
          "${device##*/}" "$vendor" "$product" "$(<"${device}/power/control")"
      fi
      ;;
  esac
done

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
  echo "WARNING: this terminal has not inherited the dialout group yet."
  echo "Run 'newgrp dialout' here, or log out and back in, before using the robot serial ports."
fi

echo "Installed persistent USB power and serial-access rules: ${RULE_TARGET}"

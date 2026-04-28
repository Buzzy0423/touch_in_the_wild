import fcntl
import os
from pathlib import Path
from typing import List, Tuple


USBDEVFS_RESET = ord("U") << 8 | 20
ELGATO_VENDOR_ID = "0fd9"


def find_elgato_usb_devices() -> List[Path]:
    devices = []
    sysfs_root = Path("/sys/bus/usb/devices")
    if not sysfs_root.exists():
        return devices

    for device_dir in sysfs_root.iterdir():
        vendor_path = device_dir / "idVendor"
        bus_path = device_dir / "busnum"
        dev_path = device_dir / "devnum"
        if not vendor_path.exists() or not bus_path.exists() or not dev_path.exists():
            continue
        vendor_id = vendor_path.read_text().strip().lower()
        if vendor_id != ELGATO_VENDOR_ID:
            continue

        busnum = int(bus_path.read_text().strip())
        devnum = int(dev_path.read_text().strip())
        devices.append(Path(f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"))

    return devices


def reset_usb_device(device_path: Path) -> None:
    fd = os.open(str(device_path), os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)


def reset_all_elgato_devices(raise_on_error: bool = False) -> List[Tuple[Path, str]]:
    failures = []
    for device_path in find_elgato_usb_devices():
        try:
            reset_usb_device(device_path)
        except OSError as exc:
            if raise_on_error:
                raise
            failures.append((device_path, str(exc)))
    return failures

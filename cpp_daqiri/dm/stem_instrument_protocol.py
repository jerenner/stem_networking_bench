"""Shared presentation of instrument observations for Qt and DigitalMicrograph."""

import datetime


TEMPERATURE_CHANNELS = ("Q1", "Q2", "Q3", "Q4", "B1", "B2", "HS1", "HS2")
BIAS_CHANNELS = (
    "DVDD18", "DVDD33", "DVDD2", "LVDSBIAS", "LVDSB_SUM", "3V3DIG", "3V3_ANA",
    "3V3_EPR", "AVDD", "VDDP", "VDDR", "VGUARD", "4V3", "2V8", "2V0", "M2V5",
    "BIASN", "BIASP", "N3BUF", "OSBUF", "BIASN_SUM", "BIASP_SUM", "N3BUF_SUM", "OSBUF_SUM",
)


def measurement_text(measurement):
    if measurement is None:
        return "not read"
    if isinstance(measurement, dict):
        value = measurement.get("value")
        if value is None:
            return "not read"
        suffix = " [suspect]" if measurement.get("valid") is False else ""
        return "{:.4g} {}{}".format(value, measurement.get("unit", ""), suffix).strip()
    return str(measurement)


def read_status(record):
    if not record:
        return "not read (legacy/mock state)"
    status = record.get("status", "not_read")
    timestamp = record.get("sampled_at")
    if timestamp is not None:
        stamp = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
        status += " / {} UTC".format(stamp.strftime("%Y-%m-%d %H:%M:%S"))
    if record.get("stale"):
        status += " / STALE"
    if record.get("error"):
        status += " / " + str(record["error"])
    return status


def link_text(module):
    if module.get("error"):
        return "ERROR: " + str(module["error"])
    masks = module.get("masks", {})
    return "{}; sync=0x{:02x}, table=0x{:02x}, disparity=0x{:02x}, K=0x{:02x}".format(
        module.get("status", "unknown"), masks.get("jesd_sync", 0),
        masks.get("not_in_table", 0), masks.get("disparity_error", 0),
        masks.get("characters_were_ks", 0),
    )


def instrument_view(instrument):
    camera = instrument.get("camera", {})
    detector = instrument.get("detector", {})
    records = instrument.get("reads", {})
    capabilities = instrument.get("capabilities", {})
    read_only = capabilities.get("read_only", instrument.get("mode") == "hardware_readonly")
    commands = capabilities.get("commands")
    if commands is None:
        commands = ["camera.read_temperature", "camera.read_biases", "detector.read_link_status"]
    online = bool(instrument.get("enabled") and instrument.get("service_online"))
    maximum = camera.get("temperature_c")
    maximum_text = "unknown" if maximum is None else "{:.4g} C".format(maximum)
    synchronized = detector.get("synchronized")
    aligned = detector.get("aligned")
    target = camera.get("target_temperature_c")
    return {
        "online": online, "mode": instrument.get("mode", "disabled"),
        "read_only": read_only, "commands": commands,
        "maximum_temperature": maximum_text,
        "target_temperature": "unknown" if target is None else "{:.4g} C".format(target),
        "temperature_warning": camera.get("temperature_warning", ""),
        "temperatures": camera.get("temperatures", {}), "biases": camera.get("biases", {}),
        "temperature_status": read_status(records.get("temperature")),
        "bias_status": read_status(records.get("biases")),
        "link_status": read_status(records.get("links")),
        "synchronized": "unknown" if synchronized is None else ("yes" if synchronized else "no"),
        "aligned": "unknown" if aligned is None else ("yes" if aligned else "no"),
        "links": detector.get("modules", []),
        "link_summary": ", ".join(str(item) for item in detector.get("links", [])) or "not read",
        "operation": instrument.get("operation", {}),
    }

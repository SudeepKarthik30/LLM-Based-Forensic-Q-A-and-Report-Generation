"""
parsers.py  —  Phase 1: Raw Artifact Parsing
─────────────────────────────────────────────────────────────────────────────
This file carries ALL the Phase-1 parsing logic (evtx / pcap / csv / log).
It is the same logic as the original parse_artifacts.py but re-packaged as
clean, importable functions so ingest.py can call them directly.

Changes from original
─────────────────────
• max_records default raised from 2 000 → 50 000 (config.MAX_RECORDS).
• TRUNCATION WARNING logged loudly (console + parser_errors.log) when cap hit.
• EVTX parser logs every skipped malformed record (record index + exception)
  for chain-of-custody auditing.
• PCAP parser logs every skipped non-IP frame (timestamp + frame type).
• CSV parser validates which timestamp/hostname columns were matched and logs
  when it falls through to "Unknown".
• ALL timestamps normalised to ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ) so
  downstream time-range queries work consistently across all formats.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import csv
import socket
import logging
import logging.handlers
from datetime import datetime, timezone, timedelta

from tqdm import tqdm
import Evtx.Evtx as evtx
import xml.etree.ElementTree as ET
import dpkt

from config import MAX_RECORDS, PARSER_LOG


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL LOGGER  (writes to parser_errors.log + console)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_parser_logger():
    """
    Configures a dedicated logger for parser skip/truncation events.

    Messages go to both the terminal (WARNING+) and to PARSER_LOG file
    (DEBUG+) so every skip is permanently recorded for chain-of-custody.
    The log file rotates at 10 MB and keeps 3 backups.
    """
    logger = logging.getLogger("forensic.parsers")
    if logger.handlers:          # avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler — captures everything
    fh = logging.handlers.RotatingFileHandler(
        PARSER_LOG, maxBytes=10 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s UTC | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))

    # Console handler — shows WARNINGs and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("   [WARNING]  %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


_log = _setup_parser_logger()


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# XML namespace used by all Windows Event Log files
NS = {'e': 'http://schemas.microsoft.com/win/2004/08/events/event'}

# Human-friendly labels for the most important Windows security EventIDs
KNOWN_EVENTS = {
    "4624": "Successful Logon",
    "4625": "Failed Logon",
    "4648": "Explicit Credential Logon",
    "4672": "Special Privileges Assigned",
    "4688": "New Process Created",
    "4698": "Scheduled Task Created",
    "4720": "User Account Created",
    "4732": "User Added to Privileged Group",
    "7045": "New Service Installed",
    "1102": "Audit Log Cleared",
}

# Ports commonly used by C2 / backdoor tools — worth flagging
SUSPICIOUS_PORTS = {4444, 1337, 31337, 6666, 9999}

# Maps IP protocol numbers to readable strings
PROTO_MAP = {
    dpkt.ip.IP_PROTO_TCP:  "TCP",
    dpkt.ip.IP_PROTO_UDP:  "UDP",
    dpkt.ip.IP_PROTO_ICMP: "ICMP",
}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_record(source_file, fmt, timestamp, hostname, event_type, data):
    """
    Builds one standardised event record — the single output format shared by
    all three parsers (evtx, pcap, csv/log).

    Think of this as stamping a "forensic report card" so every event looks the
    same regardless of where it came from.

    Inputs
    ------
    source_file  : original filename (e.g. "mimikatz.evtx")
    fmt          : format string — "evtx" | "pcap" | "csv" | "log"
    timestamp    : ISO-8601 UTC string — ALL parsers normalise to this format
    hostname     : machine / IP that generated the event
    event_type   : human label like "Failed Logon" or "Network-TCP"
    data         : dict with all parsed fields specific to this event

    Output
    ------
    dict following the Phase-1 unified schema
    """
    return {
        "source_file": os.path.basename(source_file),
        "format":      fmt,
        "timestamp":   timestamp,
        "hostname":    hostname,
        "event_type":  event_type,
        "data":        data,
    }


def _ip_to_str(raw_bytes):
    """
    Converts a 4-byte IP address (from dpkt) into a dotted-decimal string.
    Returns "unknown" if conversion fails — keeps the parser robust.

    Input  : raw bytes (4 bytes from dpkt packet)
    Output : string like "192.168.1.1"
    """
    try:
        return socket.inet_ntoa(raw_bytes)
    except Exception:
        return "unknown"


def _to_iso_utc(raw_ts, source_hint=""):
    """
    Normalises timestamps to ISO-8601 UTC.

    Supports:
    - datetime objects
    - 2014-11-24 05:07:43+00:00
    - 2014-11-24T05:07:43+00:00
    - 2014-11-24T05:07:43Z
    - 2014-11-24T05:07:43.123456Z
    - 2014-11-24 05:07:43.123456 UTC
    - MM/DD/YYYY
    - Syslog timestamps
    """

    if isinstance(raw_ts, datetime):
        if raw_ts.tzinfo is None:
            raw_ts = raw_ts.replace(tzinfo=timezone.utc)
        return raw_ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not raw_ts or raw_ts in ("Unknown", "unknown", ""):
        return "Unknown"

    raw_ts = str(raw_ts).strip()

    # --------------------------------------------------------
    # Handle ISO timestamps with timezone offsets (+00:00)
    # --------------------------------------------------------
    try:
        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    # Remove UTC marker if present
    cleaned = raw_ts.replace(" UTC", "").replace("Z", "")

    # Replace first space only
    if " " in cleaned and "T" not in cleaned:
        cleaned = cleaned.replace(" ", "T", 1)

    fmt_candidates = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%YT%H:%M:%S",
        "%m/%d/%Y",
        "%b %d %H:%M:%S",
        "%d-%m-%YT%H:%M:%S",   # NEW — matches "11-06-2026T10:59:51" (after T replace)
        "%d-%m-%Y %H:%M:%S",   # fallback
    ]

    for fmt in fmt_candidates:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    _log.warning(
        "TIMESTAMP_PARSE_FAILED | source=%s | raw=%r | kept as-is",
        source_hint,
        raw_ts,
    )

    return raw_ts

# ─────────────────────────────────────────────────────────────────────────────
# EVTX PARSER  (Windows Event Logs)
# ─────────────────────────────────────────────────────────────────────────────

def parse_evtx(filepath, max_records=None):
    """
    Parses a Windows Event Log (.evtx) file into a list of unified records.

    Each record in an evtx file is stored as XML inside the binary file.
    We extract: EventID, timestamp, hostname, channel, PID, UserSID, and all
    key-value pairs from the <EventData> section.

    Inputs
    ------
    filepath    : absolute path to .evtx file
    max_records : stop after this many events (default: config.MAX_RECORDS).
                  Set to 0 to disable the cap and process the entire file.
                  A TRUNCATION WARNING is logged loudly when the cap is hit.

    Chain of custody
    ----------------
    Every skipped malformed record is logged to parser_errors.log with
    its record index and the exception message so nothing is silently lost.

    Output
    ------
    list of event dicts (see make_record for schema)
    All timestamps are normalised to ISO-8601 UTC.
    """
    if max_records is None:
        max_records = MAX_RECORDS

    records   = []
    skipped   = 0
    basename  = os.path.basename(filepath)

    try:
        with evtx.Evtx(filepath) as log:
            for rec_idx, record in enumerate(tqdm(
                log.records(),
                desc=f"   [FILE] {basename}",
                leave=False,
                unit="ev",
            )):
                # ── Per-record parse ───────────────────────────────────────
                try:
                    root       = ET.fromstring(record.xml())
                    sys_node   = root.find('e:System',    NS)
                    edata_node = root.find('e:EventData', NS)

                    if sys_node is None:
                        _log.debug(
                            "EVTX_SKIP_NO_SYSTEM | file=%s | rec_idx=%d",
                            basename, rec_idx,
                        )
                        skipped += 1
                        continue

                    # Pull core fields from the <System> block
                    eid       = sys_node.findtext('e:EventID',  '', NS)
                    tc        = sys_node.find('e:TimeCreated', NS)
                    raw_ts    = tc.get('SystemTime', '') if tc is not None else ''
                    timestamp = _to_iso_utc(raw_ts, source_hint=basename)
                    hostname  = sys_node.findtext('e:Computer', '', NS)
                    channel   = sys_node.findtext('e:Channel',  '', NS)

                    exec_node = sys_node.find('e:Execution', NS)
                    pid       = exec_node.get('ProcessID', '') if exec_node is not None else ''

                    sec_node  = sys_node.find('e:Security', NS)
                    usid      = sec_node.get('UserID', '') if sec_node is not None else ''

                    # Flatten all <Data Name="..."> nodes from EventData
                    event_data = {}
                    if edata_node is not None:
                        for d in edata_node.findall('e:Data', NS):
                            name = d.get('Name', 'Field')
                            val  = (d.text or '').strip()
                            if val:
                                event_data[name] = val

                    records.append(make_record(
                        filepath, "evtx", timestamp, hostname,
                        KNOWN_EVENTS.get(eid, f"EventID-{eid}"),
                        {
                            "EventID":   eid,
                            "Channel":   channel,
                            "ProcessID": pid,
                            "UserSID":   usid,
                            "EventData": event_data,
                        },
                    ))

                except Exception as exc:
                    # Log every skip — chain of custody requires no silent drops
                    _log.warning(
                        "EVTX_RECORD_SKIP | file=%s | rec_idx=%d | error=%s",
                        basename, rec_idx, exc,
                    )
                    skipped += 1
                    continue

                # ── Truncation cap check ───────────────────────────────────
                if max_records and len(records) >= max_records:
                    _log.warning(
                        "TRUNCATION WARNING | file=%s | cap=%d hit — "
                        "remaining records NOT ingested. "
                        "Increase MAX_RECORDS in config.py or set to 0 to disable.",
                        basename, max_records,
                    )
                    print(
                        f"\n     TRUNCATION WARNING: {basename} hit the "
                        f"{max_records:,}-record cap. "
                        f"Records beyond this point were NOT ingested. "
                        f"Set MAX_RECORDS=0 in config.py to disable this cap."
                    )
                    break

    except Exception as e:
        _log.error("EVTX_FILE_ERROR | file=%s | error=%s", basename, e)
        print(f"   [WARNING]  EVTX parse error [{basename}]: {e}")

    if skipped:
        _log.info("EVTX_PARSE_COMPLETE | file=%s | parsed=%d | skipped=%d",
                  basename, len(records), skipped)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# PCAP PARSER  (Network Captures)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pcap(filepath, max_records=None):
    """
    Parses a network packet capture (.pcap / .pcapng) into a list of records.

    For each IP packet we extract: source/dest IPs, protocol, ports, TTL, and
    HTTP method+URI when the packet carries HTTP traffic.
    We also flag packets to/from known C2 ports (4444, 1337, 31337 …).

    Inputs
    ------
    filepath    : absolute path to .pcap or .pcapng file
    max_records : stop after this many packets (default: config.MAX_RECORDS).
                  A TRUNCATION WARNING is logged if the cap is hit.

    Chain of custody
    ----------------
    Every skipped non-IP frame (ARP, 802.1Q, etc.) is logged to
    parser_errors.log with its packet timestamp so the skip is auditable.

    Output
    ------
    list of event dicts (one per IP packet)
    All timestamps are ISO-8601 UTC.
    """
    if max_records is None:
        max_records = MAX_RECORDS

    records  = []
    skipped  = 0
    basename = os.path.basename(filepath)

    try:
        with open(filepath, 'rb') as f:
            # Try pcap first; fall back to pcapng format
            try:
                pcap = dpkt.pcap.Reader(f)
            except Exception:
                f.seek(0)
                pcap = dpkt.pcapng.Reader(f)

            for pkt_idx, (ts, buf) in enumerate(tqdm(
                pcap,
                desc=f"   [NET] {basename}",
                leave=False,
                unit="pkt",
            )):
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        # Log non-IP frames — they're skipped but not lost silently
                        frame_type = type(eth.data).__name__
                        _log.debug(
                            "PCAP_SKIP_NON_IP | file=%s | pkt_idx=%d | "
                            "ts=%.3f | frame_type=%s",
                            basename, pkt_idx, ts, frame_type,
                        )
                        skipped += 1
                        continue

                    ip      = eth.data
                    proto   = PROTO_MAP.get(ip.p, f"PROTO-{ip.p}")
                    src_ip  = _ip_to_str(ip.src)
                    dst_ip  = _ip_to_str(ip.dst)

                    data = {
                        "src_ip":        src_ip,
                        "dst_ip":        dst_ip,
                        "protocol":      proto,
                        "packet_length": len(buf),
                        "ttl":           ip.ttl,
                    }

                    # Add port numbers for TCP/UDP
                    if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                        t = ip.data
                        data["src_port"] = t.sport
                        data["dst_port"] = t.dport
                        if t.dport in SUSPICIOUS_PORTS or t.sport in SUSPICIOUS_PORTS:
                            data["alert"] = f"Suspicious C2 port: {t.dport}"

                    # Try to decode HTTP request details from TCP payload
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        try:
                            http = dpkt.http.Request(ip.data.data)
                            data["http_method"] = http.method
                            data["http_uri"]    = http.uri
                        except Exception:
                            pass    # not every TCP packet is HTTP — that's fine

                    ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    records.append(make_record(
                        filepath, "pcap", ts_str, src_ip, f"Network-{proto}", data
                    ))

                except Exception as exc:
                    _log.warning(
                        "PCAP_PACKET_SKIP | file=%s | pkt_idx=%d | error=%s",
                        basename, pkt_idx, exc,
                    )
                    skipped += 1
                    continue

                # ── Truncation cap check ───────────────────────────────────
                if max_records and len(records) >= max_records:
                    _log.warning(
                        "TRUNCATION WARNING | file=%s | cap=%d hit — "
                        "remaining packets NOT ingested.",
                        basename, max_records,
                    )
                    print(
                        f"\n     TRUNCATION WARNING: {basename} hit the "
                        f"{max_records:,}-packet cap. "
                        f"Packets beyond this point were NOT ingested."
                    )
                    break

    except Exception as e:
        _log.error("PCAP_FILE_ERROR | file=%s | error=%s", basename, e)
        print(f"   [WARNING]  PCAP parse error [{basename}]: {e}")

    if skipped:
        _log.info("PCAP_PARSE_COMPLETE | file=%s | parsed=%d | skipped=%d",
                  basename, len(records), skipped)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# CSV / LOG PARSER  (Text-based logs)
# ─────────────────────────────────────────────────────────────────────────────

# Priority-ordered column name candidates for each field
_TS_COLS       = ['TimeCreated', 'timestamp', 'Date', 'DateTime', 'time',
                   'EventTime', 'date_time']
_HOSTNAME_COLS = ['MachineName', 'hostname', 'Computer', 'Host',
                   'machine', 'computer_name']
                   # NOTE: system_logs.csv does not contain any hostname column, so it will fall back to 'Unknown'.
_ETYPE_COLS    = ['LevelDisplayName', 'Id', 'EventId', 'event_type',
                   'level', 'severity']


def _pick_col(row, candidates):
    """
    Picks the first matching column from `candidates` that exists in `row`
    and has a non-empty value.  Returns (column_name, value) or (None, None).
    """
    for col in candidates:
        val = row.get(col, '').strip()
        if val:
            return col, val
    return None, None


def parse_csv_log(filepath, max_records=None):
    """
    Parses a CSV export or plain-text log file (syslog style).

    For CSV files we look for common column names to find timestamp, hostname,
    and event type.  For plain text we split on whitespace and grab the first
    few tokens as timestamp + hostname.

    Inputs
    ------
    filepath    : absolute path to .csv or .log file
    max_records : stop after this many rows/lines (default: config.MAX_RECORDS)

    Column matching
    ---------------
    Tries multiple candidate column names in priority order (see _TS_COLS etc.).
    Logs which column was matched for each field, and logs a WARNING if a field
    falls through to "Unknown" so silent column mismatches are caught.

    Timestamps
    ----------
    All timestamps are normalised to ISO-8601 UTC via _to_iso_utc().

    Output
    ------
    list of event dicts
    """
    if max_records is None:
        max_records = MAX_RECORDS

    records  = []
    basename = os.path.basename(filepath)
    ext      = os.path.splitext(filepath)[1].lower()

    try:
        if ext == '.csv':
            with open(filepath, newline='', encoding='utf-8', errors='replace') as f:
                reader  = csv.DictReader(f)
                headers = reader.fieldnames or []
                _log.info("CSV_OPEN | file=%s | columns=%s", basename, headers)

                for row_idx, row in enumerate(reader):
                    if max_records and row_idx >= max_records:
                        _log.warning(
                            "TRUNCATION WARNING | file=%s | cap=%d hit — "
                            "remaining rows NOT ingested.",
                            basename, max_records,
                        )
                        print(
                            f"\n     TRUNCATION WARNING: {basename} hit the "
                            f"{max_records:,}-row cap."
                        )
                        break

                    ts_col,   raw_ts   = _pick_col(row, _TS_COLS)
                    host_col, hostname = _pick_col(row, _HOSTNAME_COLS)
                    etype_col, etype   = _pick_col(row, _ETYPE_COLS)

                    # Log column resolution results on first row only
                    if row_idx == 0:
                        _log.info(
                            "CSV_COL_MATCH | file=%s | "
                            "ts_col=%s | host_col=%s | etype_col=%s",
                            basename, ts_col, host_col, etype_col,
                        )

                    # Warn on fall-throughs to Unknown
                    if ts_col is None:
                        _log.warning(
                            "CSV_COL_UNKNOWN | file=%s | row=%d | "
                            "field=timestamp — none of %s found in headers",
                            basename, row_idx, _TS_COLS,
                        )
                    if host_col is None:
                        _log.warning(
                            "CSV_COL_UNKNOWN | file=%s | row=%d | "
                            "field=hostname — none of %s found in headers",
                            basename, row_idx, _HOSTNAME_COLS,
                        )

                    timestamp = _to_iso_utc(raw_ts or 'Unknown', source_hint=basename)
                    hostname  = hostname or 'Unknown'
                    etype     = etype   or 'Log-Entry'

                    records.append(make_record(
                        filepath, "csv", timestamp, hostname, str(etype), dict(row)
                    ))

        else:
            # Plain syslog-style: "Month Day HH:MM:SS hostname process: message"
            with open(filepath, encoding='utf-8', errors='replace') as f:
                for row_idx, line in enumerate(f):
                    if max_records and row_idx >= max_records:
                        _log.warning(
                            "TRUNCATION WARNING | file=%s | cap=%d hit.",
                            basename, max_records,
                        )
                        break

                    line = line.strip()
                    if not line:
                        continue

                    parts    = line.split(' ', 5)
                    raw_ts   = ' '.join(parts[:3]) if len(parts) >= 3 else 'Unknown'
                    hostname = parts[3]             if len(parts) >  3 else 'Unknown'
                    timestamp = _to_iso_utc(raw_ts, source_hint=basename)

                    records.append(make_record(
                        filepath, "log", timestamp, hostname,
                        "Syslog-Entry", {"raw_line": line},
                    ))

    except Exception as e:
        _log.error("CSV_FILE_ERROR | file=%s | error=%s", basename, e)
        print(f"   [WARNING]  CSV/LOG parse error [{basename}]: {e}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PARSE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def parse_all_files(input_dir, output_dir):
    """
    Scans `input_dir` for every supported forensic file and parses them all.

    Supported formats: .evtx, .pcap, .pcapng, .csv, .log

    After parsing, each file gets its own JSON in `output_dir`, plus a combined
    `all_artifacts.json` containing every event from every file.

    Known limitation
    ----------------
    If MAX_RECORDS (config.py) is set, files larger than that cap will be
    truncated. A TRUNCATION WARNING is printed/logged for every truncated file.
    Set MAX_RECORDS=0 in config.py to disable all caps and process entire files
    (may require significant RAM for very large datasets).

    Inputs
    ------
    input_dir  : folder containing raw forensic artifacts (e.g. sample_data/)
    output_dir : folder where parsed JSON files will be written (e.g. output/)

    Output
    ------
    list of ALL event dicts from every parsed file (combined)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map each extension to the right parser function
    ext_map = {
        '.evtx':   parse_evtx,
        '.pcap':   parse_pcap,
        '.pcapng': parse_pcap,
        '.csv':    parse_csv_log,
        '.log':    parse_csv_log,
    }

    # Collect only files we know how to parse
    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in ext_map
    ]

    if not files:
        print("   [WARNING]  No supported files found! Check that sample_data/ has .evtx/.pcap/.csv files.")
        return []

    all_records = []

    for fname in files:
        ext    = os.path.splitext(fname)[1].lower()
        fpath  = os.path.join(input_dir, fname)
        parser = ext_map[ext]

        records  = parser(fpath)
        out_path = os.path.join(output_dir, fname.replace(ext, '.json'))

        # Save individual JSON for this file (same as original Phase 1 behaviour)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

        print(f"   [OK]  {fname:<50}  →  {len(records):>5,} events")
        all_records.extend(records)

    # Also write the big combined file
    combined_path = os.path.join(output_dir, "all_artifacts.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)

    return all_records

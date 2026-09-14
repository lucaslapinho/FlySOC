"""Deterministic SOC-like telemetry fixtures. All command strings are inert data."""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

# Family, generic rule name, process, command template, category, technique,
# default verdict. Similar rules intentionally occur in different families.
PROFILES = [
    ("powershell_admin", "PowerShell activity", "powershell.exe", "powershell inventory.ps1 -Host host{host} -Report inventory{variant}.csv", "execution", "T1059.001", "FALSE_POSITIVE"),
    ("powershell_encoded", "PowerShell activity", "powershell.exe", "powershell -EncodedCommand [SYNTHETIC_ENCODED_{variant}] -WindowStyle Hidden", "execution", "T1059.001", "TRUE_POSITIVE"),
    ("rdp_admin", "Remote logon", "mstsc.exe", "mstsc /v:host{host} change-ticket CHG{variant} helpdesk", "remote_access", "T1021.001", "BENIGN"),
    ("rdp_lateral", "Remote logon", "mstsc.exe", "mstsc /v:host{host} offhours unmanaged origin session{variant}", "remote_access", "T1021.001", "TRUE_POSITIVE"),
    ("brute_force", "Authentication failures", "sshd.exe", "authentication failure repeated password attempts user{user} count {count}", "authentication", "T1110.001", "TRUE_POSITIVE"),
    ("password_spray", "Authentication failures", "auth-service", "authentication failures distributed accounts low attempts spray campaign{variant}", "authentication", "T1110.003", "TRUE_POSITIVE"),
    ("privileged_admin", "Administrative activity", "net.exe", "privileged group membership change user{user} administrative maintenance CHG{variant}", "administration", "T1078", "FALSE_POSITIVE"),
    ("vuln_scanner", "Network discovery", "scanner-agent", "approved vulnerability assessment subnet 10.20.{variant}.0 profile scheduled", "discovery", "T1046", "FALSE_POSITIVE"),
    ("backup", "Data transfer", "backup-agent.exe", "backup-agent snapshot volume weekly retention job{variant}", "collection", "T1005", "BENIGN"),
    ("software_deployment", "Software installation", "msiexec.exe", "msiexec /i corporate-package{variant}.msi /quiet approved deployment", "execution", "T1072", "BENIGN"),
    ("service_account", "Authentication failures", "service-host.exe", "service authentication expired credential svc_backup retry job{variant}", "authentication", "T1078", "FALSE_POSITIVE"),
    ("scheduled_task", "Scheduled task activity", "task-scheduler", "scheduled task unapproved hidden persistence user{user} task{variant}", "persistence", "T1053.005", "TRUE_POSITIVE"),
    ("remote_admin", "Remote logon", "remote-support.exe", "remote-support approved session host{host} helpdesk ticket{variant}", "remote_access", "T1021", "FALSE_POSITIVE"),
    ("benign_script", "PowerShell activity", "powershell.exe", "powershell healthcheck.ps1 -Service monitoring -Log status{variant}.txt", "execution", "T1059.001", "BENIGN"),
]
HELDOUT = ("credential_dumping", "Sensitive process access", "memory-tool.exe",
           "[SIMULATED] lsass memory access credential dump process snapshot handle{variant}",
           "credential_access", "T1003.001", "TRUE_POSITIVE")
VERDICTS = np.array(["TRUE_POSITIVE", "FALSE_POSITIVE", "BENIGN"])


def generate_alerts(cfg: dict[str, Any], size: int | None = None) -> pd.DataFrame:
    """Generate temporal recurrence plus a whole unseen family only in test.

    Stochastic labels and overlapping severity prevent a perfectly deterministic
    verdict lookup. Ground-truth family is for evaluation only.
    """
    n = int(size if size is not None else cfg["dataset"]["size"])
    if n < 100:
        raise ValueError("Generate at least 100 alerts")
    if cfg["dataset"]["heldout_family"] != HELDOUT[0]:
        raise ValueError("This generator currently holds out credential_dumping")
    rng = np.random.default_rng(cfg["seed"])
    start_test = int(n * (cfg["evaluation"]["train_ratio"] + cfg["evaluation"]["validation_ratio"]))
    test_count = n - start_test
    unseen_count = max(1, min(test_count - 1, round(test_count * cfg["dataset"]["unseen_test_fraction"])))
    unseen_rows = set(rng.choice(np.arange(start_test, n), unseen_count, replace=False).tolist())
    origin = pd.Timestamp(cfg["dataset"]["start"]).to_pydatetime()
    rows = []
    for i in range(n):
        unseen = i in unseen_rows
        profile = HELDOUT if unseen else PROFILES[int(rng.integers(len(PROFILES)))]
        family, name, process, template, category, technique, typical_verdict = profile
        host, user, variant = int(rng.integers(1, 81)), int(rng.integers(1, 121)), int(rng.integers(1, 13))
        verdict = str(rng.choice(VERDICTS)) if rng.random() < .07 else typical_verdict
        command = template.format(host=host, user=user, variant=variant, count=int(rng.integers(4, 150)))
        if rng.random() < .3:
            command += f" trace_id={int(rng.integers(1000, 99999))}"
        if unseen and rng.random() < .5:
            process = "process-snapshot.exe"
            command = f"[SIMULATED] credential material lsass.exe memory minidump snapshot handle{variant}"
        row = {
            "timestamp": (origin + timedelta(minutes=i * 7, seconds=int(rng.integers(50)))).isoformat(),
            "alert_id": f"ALERT{i + 1:06d}", "alert_name": name,
            "rule_id": f"RULE-{category.upper()}", "severity": str(rng.choice(["low", "medium", "high"], p=[.2, .5, .3])),
            "category": category, "mitre_tactic": category, "mitre_technique": technique,
            "hostname": f"host{host:03d}", "username": f"svc_{user % 5}" if rng.random() < .25 else f"user{user:03d}",
            "source_ip": f"10.20.{host % 8}.{int(rng.integers(1, 255))}",
            "destination_ip": f"10.30.{host % 5}.{int(rng.integers(1, 255))}",
            "source_port": int(rng.integers(49152, 65536)),
            "destination_port": 3389 if category == "remote_access" else int(rng.choice([22, 80, 443, 445, 5985])),
            "process_name": process, "process_command_line": command,
            "action": str(rng.choice(["detected", "allowed", "blocked"], p=[.65, .25, .1])),
            "verdict": verdict, "family": family, "is_unseen": unseen,
        }
        for field in ("username", "source_ip", "destination_port", "process_command_line", "mitre_technique"):
            if rng.random() < .035:
                row[field] = None
        rows.append(row)
    return pd.DataFrame(rows)

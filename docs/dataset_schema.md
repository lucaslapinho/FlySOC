# Dataset schema

FlySOC reads local `.csv` (comma separated) and `.tsv` (tab separated) files.
UTF-8 is recommended. All telemetry, including shell commands, URLs and paths,
is handled strictly as data. No command is executed, URL fetched or referenced
file opened from a telemetry cell.

| Field | Type / role | Missing-value behavior |
|---|---|---|
| `timestamp` | Parseable timestamp; timezone-aware recommended | Required and valid for chronological evaluation; optional for encoding alone |
| `alert_id` | Unique historical identifier | Generated as row IDs if column absent; null/duplicate IDs rejected |
| `alert_name` | Text feature and display name | Empty text |
| `rule_id` | Metadata | Ignored by features |
| `severity` | Categorical feature | Token omitted |
| `category` | Categorical feature | Token omitted |
| `mitre_tactic`, `mitre_technique` | Metadata | Ignored by default feature configuration |
| `hostname`, `username` | Low-weight hashed identity | Token omitted |
| `source_ip`, `destination_ip` | Low-weight hashed identity | Token omitted; never contacted |
| `source_port`, `destination_port` | Numeric 0–65535 | Invalid values zeroed and presence flag cleared |
| `process_name` | Categorical feature | Token omitted |
| `process_command_line` | Text feature | Empty text; never interpreted as executable code |
| `action` | Categorical feature | Token omitted |
| `verdict` | Optional supervision label | Unlabeled rows retained in memory; excluded from supervised scoring |
| `metadata` | Optional additional data | Preserved in historical metadata; excluded from features |

Accepted supervised verdicts are `TRUE_POSITIVE`, `FALSE_POSITIVE` and `BENIGN`.
File ingestion trims whitespace and uppercases verdicts. Other values are
unlabeled for classification. `UNKNOWN` is accepted by the feedback API.
Classification is skipped if training has fewer than two observed classes.

The browser Alert Analyzer also accepts JSON for one-off encoding. Use a flat
object, an array of flat objects, or `{ "alerts": [...] }`. CSV and TSV imports
need a header row; each data row becomes one selectable alert. Browser import is
limited to 5 MB and 1,000 rows, while the selected alert sent to the loopback
service must fit its 64 KiB request limit. Nested JSON values are rejected.
These browser limits do not replace the stricter timestamp and identifier rules
used when an entire dataset is trained or evaluated chronologically.

Synthetic data also contains `family` and `is_unseen`. These are **evaluation
ground truth only**, never features. Actual unseen status is calculated from
families absent in training, rather than trusting the supplied `is_unseen` flag.
Ranking and cluster-purity evaluation require a complete `family` column. An
external export without this annotation can still be encoded and scored, but
has no measured semantic ranking quality or false-merge truth.

Text is capped at 16,384 characters per field during feature extraction;
categorical values are capped at 512. Original metadata remains available.
Missing optional columns, nulls, unfamiliar categories and high-cardinality
identities are supported. Empty feature rows receive empty fingerprints and
should be investigated as data quality problems before interpreting novelty.

## Synthetic patterns

Fourteen historical families cover administrative and encoded PowerShell,
administrative/lateral RDP, brute force, password spraying, privileged activity,
vulnerability scanning, backup, deployment, service accounts, scheduled tasks,
remote support and benign scripts. A fifteenth family, credential dumping with
LSASS-related variants, is injected only into the final test period.

Each family recurs with changes to hosts, identities, ports, trace IDs and missing
values. Severity is sampled independently of verdict. About 7% of rows draw a
random verdict from all three labels, which includes the possibility of drawing
the original label. Therefore the effective changed-label fraction is below 7%.

## Future Cortex exports

Map vendor export columns to this schema in an adapter; do not assume a Cortex
tenant contains these exact names. Exports for alerts, cases, `xdr_data`,
`management_auditing` and analyst verdicts may need distinct mappings and joins.
Export-specific timestamps, verdict meanings and one-to-many relationships must
be validated before evaluation. No production API integration is present.

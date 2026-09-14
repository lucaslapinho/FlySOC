# Security policy

FlySOC is experimental, local research software. It does not provide automatic
incident response or production alert suppression. The maintained development
line is currently v0.2.

The Brain Lab server binds to loopback. Keep it local. Telemetry is data and
must never be executed. Joblib models are executable serialization: use only
trusted models you generated or have independently verified. A matching checksum
establishes integrity against a manifest, not trust in its author.

For a potential vulnerability, use the repository's private **Report a
vulnerability** option when available. If it is unavailable, open an issue asking
for a private reporting channel without including exploit details, secrets or
real telemetry. Please include affected versions and a synthetic reproduction
once a private channel is established.

Ordinary numerical bugs and documentation corrections may be filed as public
issues using synthetic examples.

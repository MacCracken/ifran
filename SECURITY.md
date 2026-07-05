# Security Policy — ifran

## Threat model

ifran is a single-operator control plane: it executes **operator-authored**
job specs and manages artifacts/corpora on the operator's box. The trust
boundaries, and their defenses:

1. **The operator key** (`~/.ifran/operator.sk`) — generated from
   `getrandom(2)` via sigil `ed25519_keypair`; written 0600; `keys init`
   refuses to overwrite. Key custody is the operator's.
2. **Model artifacts** (`store add` — potentially foreign files): structural
   validation is **tula's** (its own frozen, fuzzed surface — garbage is
   rejected before anything else looks at it); authenticity is Ed25519 with an
   honest per-artifact status (`verified` / `signed-unknown-key` /
   `unsigned`); identity is sha256 content-addressing, re-derived by
   `store verify` (bit-rot/tamper detection).
3. **Job execution** — specs name an ABSOLUTE binary path; argv passes to
   `execve` directly (**no shell**, so no injection through args); child
   output is captured into a bounded buffer (4 MB, truncation marked); stderr
   merged. `timeout_s` (2.1) bounds a hung child: poll-paced capture, SIGKILL
   on expiry, recorded as `timed-out` — and `ifran reap` un-sticks `running`
   rows orphaned by a killed ifran (signal-0 PID probe; conservative — a live
   PID is never touched).
4. **The database** — all writes go through patra prepared statements with
   bound parameters (free text never re-parsed as SQL, per patra's contract).
5. **Datasets / preference text** — stored as bytes (content-addressed) or via
   bound parameters; the JSONL export escapes via the stdlib JSON escaper
   (parse-back proven in the suite).
6. **Metric extraction** — scans a bounded, operator-initiated run log for a
   numeric token; no evaluation of log content.

Out of scope: sigil/tula/patra internals (their own audit surfaces);
multi-tenant isolation (deferred by design — single-operator box first;
aegis/kavach seams when it opens); adversarial *model semantics* (the serving
layer's policy question).

## Reporting

Open an issue at <https://github.com/MacCracken/ifran/issues>, or contact the
maintainer privately for sensitive reports.

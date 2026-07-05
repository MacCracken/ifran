# ifran — Stability Policy (2.x)

**The 2.0.0 port cut froze the public surface** documented in
[`docs/api.md`](docs/api.md): CLI commands + semantics (exit codes ARE the
gates), spec formats, workspace layout, table schemas, export formats, and the
listed module functions.

- **Frozen:** everything above. Schema changes ship with in-place migrations
  (the M4 `ALTER TABLE` precedent — an existing `ifran.db` keeps working).
- **Additive-only minors:** new commands, new spec keys (absent = old
  behavior), new columns (defaults), the named additive lane in api.md.
- **Out of freeze:** `_`-internals, caps/defaults, log text.
- **`rust-old/`** is reference-only, retained (with its docs and original
  AGPL license) for a release or two past 2.0.0 before removal; it is not API.

Breaking any frozen item requires 3.0.0 with a CHANGELOG **Breaking** section
and a migration note.

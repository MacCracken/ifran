# Dependency Watch

Pinned versions, known conflicts, and upgrade paths for ifran dependencies.

---

## Pinned Versions

### redis — resolved

~~redis 0.27.6 was pinned due to majra 1.0.1 depending on `redis 0.27`.~~

**Resolved in majra 1.0.2**: `redis` upgraded to 1.x. Both ifran and majra now use `redis 1.x`. No version conflict.

---

### sha2 0.10 (held)

**Reason:** `hmac 0.13` (required for `sha2 0.11`) is pre-release. Upgrading sha2 without a stable hmac would introduce a pre-release transitive dependency.

**Upgrade path:** Wait for `hmac 0.13` stable release, then bump `sha2` to 0.11 and `hmac` to 0.13 together.

---

### rusqlite 0.39 / libsqlite3-sys 0.37

**Reason:** `rusqlite 0.39` bundles `libsqlite3-sys 0.37`. The `sqlx` crate (any version) bundles `libsqlite3-sys 0.28`, causing a Cargo `links` conflict. This is why the `postgres` feature uses `tokio-postgres`/`deadpool-postgres` instead of `sqlx`.

**Upgrade path:** If sqlx upgrades to use `libsqlite3-sys 0.37+`, we could switch to sqlx for PostgreSQL. Alternatively, if rusqlite updates to match sqlx's libsqlite3-sys version, the conflict resolves. Until then, `tokio-postgres` is the correct choice.

**See also:** Roadmap item 1.0.1 (feature-gate rusqlite behind `sqlite` feature).

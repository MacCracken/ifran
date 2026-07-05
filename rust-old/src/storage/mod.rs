pub mod cache;
#[cfg(feature = "sqlite")]
pub mod db;
pub mod encryption;
pub mod layout;
pub mod traits;

#[inline]
pub fn deserialize_quoted<T: serde::de::DeserializeOwned>(s: &str) -> serde_json::Result<T> {
    let mut quoted = String::with_capacity(s.len() + 2);
    quoted.push('"');
    quoted.push_str(s);
    quoted.push('"');
    serde_json::from_str(&quoted)
}

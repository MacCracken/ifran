//! Download progress tracking and events.
//!
//! Uses a tokio broadcast channel so multiple consumers (CLI progress bar,
//! API SSE stream, desktop UI) can subscribe to download progress.

use ifran_types::registry::DownloadState;
use tokio::sync::broadcast;

/// A single progress event emitted during a download.
#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub model_name: String,
    pub state: DownloadState,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub speed_bytes_per_sec: u64,
    pub message: Option<String>,
}

impl ProgressEvent {
    /// Progress as a percentage (0.0–100.0), or None if total is unknown.
    pub fn percent(&self) -> Option<f64> {
        self.total_bytes.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.downloaded_bytes as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Sender/receiver pair for progress events.
pub struct ProgressTracker {
    tx: broadcast::Sender<ProgressEvent>,
}

impl ProgressTracker {
    /// Create a new tracker with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to progress events.
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.tx.subscribe()
    }

    /// Send a progress event. Returns Ok even if no receivers are listening.
    pub fn send(&self, event: ProgressEvent) {
        let _ = self.tx.send(event);
    }

    /// Convenience: send a state-change event with a message.
    pub fn emit(&self, model_name: &str, state: DownloadState, message: &str) {
        self.send(ProgressEvent {
            model_name: model_name.to_string(),
            state,
            downloaded_bytes: 0,
            total_bytes: None,
            speed_bytes_per_sec: 0,
            message: Some(message.to_string()),
        });
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_percent() {
        let event = ProgressEvent {
            model_name: "test".into(),
            state: DownloadState::Downloading,
            downloaded_bytes: 50,
            total_bytes: Some(100),
            speed_bytes_per_sec: 0,
            message: None,
        };
        assert_eq!(event.percent(), Some(50.0));
    }

    #[test]
    fn progress_percent_unknown_total() {
        let event = ProgressEvent {
            model_name: "test".into(),
            state: DownloadState::Downloading,
            downloaded_bytes: 50,
            total_bytes: None,
            speed_bytes_per_sec: 0,
            message: None,
        };
        assert_eq!(event.percent(), None);
    }

    #[tokio::test]
    async fn broadcast_works() {
        let tracker = ProgressTracker::new(16);
        let mut rx = tracker.subscribe();
        tracker.emit("model", DownloadState::Queued, "starting");
        let event = rx.recv().await.unwrap();
        assert_eq!(event.model_name, "model");
        assert_eq!(event.state, DownloadState::Queued);
    }
}

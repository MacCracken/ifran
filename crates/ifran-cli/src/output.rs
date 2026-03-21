//! Terminal output formatting utilities for consistent CLI presentation.

use owo_colors::OwoColorize;

/// Print a section header (bold, cyan).
pub fn header(text: &str) {
    eprintln!("{}", text.bold().cyan());
}

/// Print a key-value pair with the key dimmed.
pub fn kv(key: &str, value: &dyn std::fmt::Display) {
    eprintln!("  {}: {}", key.dimmed(), value);
}

/// Print a success message (green checkmark).
pub fn success(msg: &str) {
    eprintln!("{} {}", "✓".green().bold(), msg);
}

/// Print a warning message (yellow).
pub fn warn(msg: &str) {
    eprintln!("{} {}", "!".yellow().bold(), msg);
}

/// Print an error message (red).
pub fn error(msg: &str) {
    eprintln!("{} {}", "✗".red().bold(), msg);
}

/// Print a dimmed info line.
pub fn info(msg: &str) {
    eprintln!("{}", msg.dimmed());
}

/// Format a byte count as human-readable size.
pub fn format_size(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    }
}

/// Truncate a string to max width, adding ellipsis if needed.
pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

/// A simple table printer with column alignment.
pub struct Table {
    headers: Vec<String>,
    widths: Vec<usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(headers: Vec<&str>) -> Self {
        let widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        Self {
            headers: headers.into_iter().map(String::from).collect(),
            widths,
            rows: Vec::new(),
        }
    }

    pub fn add_row(&mut self, cells: Vec<String>) {
        for (i, cell) in cells.iter().enumerate() {
            if i < self.widths.len() {
                self.widths[i] = self.widths[i].max(cell.len());
            }
        }
        self.rows.push(cells);
    }

    pub fn print(&self) {
        // Header
        let header_line: Vec<String> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<width$}", h, width = self.widths[i]))
            .collect();
        println!("{}", header_line.join("  ").bold());

        // Separator
        let total_width: usize = self.widths.iter().sum::<usize>() + (self.widths.len() - 1) * 2;
        println!("{}", "─".repeat(total_width).dimmed());

        // Rows
        for row in &self.rows {
            let line: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = self.widths.get(i).copied().unwrap_or(cell.len());
                    format!("{:<width$}", cell, width = w)
                })
                .collect();
            println!("{}", line.join("  "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_gigabytes() {
        assert_eq!(format_size(4_000_000_000), "4.0 GB");
        assert_eq!(format_size(1_500_000_000), "1.5 GB");
    }

    #[test]
    fn format_size_megabytes() {
        assert_eq!(format_size(500_000_000), "500 MB");
        assert_eq!(format_size(1_000_000), "1 MB");
    }

    #[test]
    fn format_size_zero() {
        assert_eq!(format_size(0), "0 MB");
    }

    #[test]
    fn truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_long() {
        let result = truncate("this is a very long model name", 10);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn table_basic() {
        let mut t = Table::new(vec!["NAME", "SIZE"]);
        t.add_row(vec!["model-a".into(), "4.0 GB".into()]);
        t.add_row(vec!["model-b".into(), "1.5 GB".into()]);
        // Just verify it doesn't panic
        t.print();
    }

    #[test]
    fn table_auto_widths() {
        let mut t = Table::new(vec!["A", "B"]);
        t.add_row(vec!["short".into(), "x".into()]);
        t.add_row(vec!["a much longer cell value".into(), "y".into()]);
        assert_eq!(t.widths[0], "a much longer cell value".len());
    }
}

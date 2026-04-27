//! `lethe transcript <path>` — extract a user/assistant turn from a
//! Claude Code JSONL transcript. Replaces the Python helpers the plugin
//! used (`scripts/transcript.py` and the inline `python3 -` block in
//! `hooks/parse-transcript.sh`) so the plugin stays Python-free.
//!
//! Output formats — preserve byte-for-byte what the previous Python
//! helpers emitted, since hook scripts grep their stdout:
//!
//! Default (no `--turn`): hook-pipeline form, includes session/turn ids.
//!     SESSION_ID: <id>
//!     TURN_ID: <id>
//!     ---
//!     USER:
//!     <text>
//!     ---
//!     ASSISTANT:
//!     <text>
//!
//! With `--turn <uuid>`: human-readable form for the recall skill.
//!     USER:
//!     <text>
//!
//!     ASSISTANT:
//!     <text>

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

pub fn run(path: &str, turn: Option<&str>) -> Result<i32> {
    let p = Path::new(path);
    if !p.exists() {
        eprintln!("transcript not found: {path}");
        return Ok(1);
    }
    let parsed = parse(p).with_context(|| format!("read {path}"))?;
    if parsed.user.is_none() && parsed.assistant.is_none() {
        eprintln!("(no matching turn)");
        return Ok(1);
    }

    if let Some(turn_id) = turn {
        let pair = pair_for_turn(p, turn_id)?;
        if pair.user.is_none() && pair.assistant.is_none() {
            eprintln!("(no matching turn)");
            return Ok(1);
        }
        println!("USER:");
        println!("{}", pair.user.unwrap_or_default().trim());
        println!();
        println!("ASSISTANT:");
        println!("{}", pair.assistant.unwrap_or_default().trim());
    } else {
        println!("SESSION_ID: {}", parsed.session_id.unwrap_or_default());
        println!("TURN_ID: {}", parsed.user_id.unwrap_or_default());
        println!("---");
        println!("USER:");
        println!("{}", parsed.user.unwrap_or_default().trim());
        println!("---");
        println!("ASSISTANT:");
        println!("{}", parsed.assistant.unwrap_or_default().trim());
    }
    Ok(0)
}

#[derive(Default)]
struct Parsed {
    session_id: Option<String>,
    user_id: Option<String>,
    user: Option<String>,
    assistant: Option<String>,
}

/// Walk the JSONL once tracking the last user prompt and the assistant
/// reply that followed it.
fn parse(path: &Path) -> Result<Parsed> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Parsed::default();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let role = role_of(&rec);
        let msg = rec.get("message").unwrap_or(&rec);
        match role.as_deref() {
            Some("user") => {
                if is_tool_result_only(msg) {
                    continue;
                }
                let text = message_text(msg, false);
                if text.is_empty() {
                    continue;
                }
                out.user = Some(text);
                out.user_id = rec
                    .get("uuid")
                    .or_else(|| rec.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
                out.session_id = rec
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .or(out.session_id);
                out.assistant = None;
            }
            Some("assistant") => {
                let text = message_text(msg, false);
                if text.is_empty() {
                    continue;
                }
                out.assistant = Some(text);
                out.session_id = rec
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .or(out.session_id);
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Find a specific turn by its UUID (matches the user's `uuid`/`id`),
/// then capture the next assistant message.
fn pair_for_turn(path: &Path, target: &str) -> Result<Parsed> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Parsed::default();
    let mut captured = false;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let role = role_of(&rec);
        let msg = rec.get("message").unwrap_or(&rec);
        let uid = rec
            .get("uuid")
            .or_else(|| rec.get("id"))
            .and_then(|v| v.as_str());
        match role.as_deref() {
            Some("user") => {
                if is_tool_result_only(msg) {
                    continue;
                }
                let text = message_text(msg, false);
                if text.is_empty() {
                    continue;
                }
                if uid == Some(target) {
                    out.user = Some(text);
                    out.user_id = uid.map(str::to_owned);
                    out.assistant = None;
                    captured = true;
                }
            }
            Some("assistant") if captured && out.assistant.is_none() => {
                let text = message_text(msg, false);
                if !text.is_empty() {
                    out.assistant = Some(text);
                    return Ok(out);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn role_of(rec: &Value) -> Option<String> {
    rec.get("message")
        .and_then(|m| m.get("role"))
        .and_then(|v| v.as_str())
        .or_else(|| rec.get("type").and_then(|v| v.as_str()))
        .map(str::to_owned)
}

fn is_tool_result_only(msg: &Value) -> bool {
    let Some(content) = msg.get("content").and_then(|c| c.as_array()) else {
        return false;
    };
    if content.is_empty() {
        return false;
    }
    content.iter().all(|b| {
        b.get("type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "tool_result")
    })
}

fn message_text(msg: &Value, include_tool_results: bool) -> String {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return s.to_owned();
    }
    let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        let Some(btype) = b.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        match btype {
            "text" => {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    parts.push(t.to_owned());
                }
            }
            "tool_use" => {
                let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                parts.push(format!("[tool_use: {name}]"));
            }
            "tool_result" if include_tool_results => {
                parts.push("[tool_result]".to_owned());
            }
            _ => {}
        }
    }
    parts.join("\n")
}

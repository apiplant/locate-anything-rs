//! The prompt templates LocateAnything-3B was trained on, one per task.
//!
//! The CLI's `--task` flag and the library share these, so a prompt built here is exactly what
//! `locate-anything --task <name>` sends.

/// Open-vocabulary detection: every instance of each category.
pub fn detect<S: AsRef<str>>(categories: &[S]) -> String {
    let cats: Vec<&str> = categories.iter().map(|c| c.as_ref().trim()).filter(|c| !c.is_empty()).collect();
    format!("Locate all the instances that matches the following description: {}.", cats.join("</c>"))
}

/// Phrase grounding: all instances matching a referring expression.
pub fn ground(description: &str) -> String {
    format!("Locate all the instances that match the following description: {description}.")
}

/// Phrase grounding: a single instance.
pub fn ground_single(description: &str) -> String {
    format!("Locate a single instance that matches the following description: {description}.")
}

/// Locate a piece of text.
pub fn text(text: &str) -> String {
    format!("Please locate the text referred as {text}.")
}

/// Scene-text detection (no query).
pub fn detect_text() -> String {
    "Detect all the text in box format.".into()
}

/// GUI element grounding, as a box.
pub fn gui(description: &str) -> String {
    format!("Locate the region that matches the following description: {description}.")
}

/// Pointing (and GUI grounding as a point).
pub fn point(description: &str) -> String {
    format!("Point to: {description}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_joins_categories() {
        assert_eq!(
            detect(&["person", " car ", ""]),
            "Locate all the instances that matches the following description: person</c>car."
        );
    }
}

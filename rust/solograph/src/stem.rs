//! Snowball stemmers, matched against Python's snowballstemmer 3.1.1.
//!
//! Russian: rust-stemmers, identical to Python on 15,656 words from the real KB
//! and session index. English: rust-stemmers implements Snowball 2, and 24 of
//! ~13,000 English words stem differently from Snowball 3.1 ("organization"
//! organ/organiz) — see english.rs for the port that closes that gap.

use std::sync::LazyLock;

use rust_stemmers::{Algorithm, Stemmer};

static RU: LazyLock<Stemmer> = LazyLock::new(|| Stemmer::create(Algorithm::Russian));

pub fn russian(word: &str) -> String {
    RU.stem(word).into_owned()
}

pub fn english(word: &str) -> String {
    crate::english::stem(word)
}

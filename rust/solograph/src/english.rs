//! English (Porter2) stemmer, Snowball 3.1.1 — a port of the code snowballstemmer
//! generates, so a query stems the same here as in Python.
//!
//! rust-stemmers implements Snowball 2, and on 13k English words from the real
//! KB 24 stemmed differently ("organization": organ vs organiz, "added": ad vs
//! add, "university": univers vs universiti). A different stem is a different
//! prefix query, so Rust and Python would answer the same search differently.
//!
//! `among` tables have no routines attached in this algorithm, so each lookup is
//! "the longest entry that matches here", which is what Snowball's binary search
//! with substring links computes.

type Among = &'static [(&'static str, i32)];

const A0: Among = &[
    ("arsen", -1),
    ("commun", -1),
    ("emerg", -1),
    ("gener", -1),
    ("inter", -1),
    ("later", -1),
    ("organ", -1),
    ("past", -1),
    ("univers", -1),
];
const A1: Among = &[("'", 1), ("'s'", 1), ("'s", 1)];
const A2: Among = &[
    ("ied", 2),
    ("s", 3),
    ("ies", 2),
    ("sses", 1),
    ("ss", -1),
    ("us", -1),
];
const A3: Among = &[("succ", 1), ("proc", 1), ("exc", 1)];
const A4: Among = &[
    ("even", 2),
    ("cann", 2),
    ("inn", 2),
    ("earr", 2),
    ("herr", 2),
    ("out", 2),
    ("y", 1),
];
const A5: Among = &[
    ("", -1),
    ("ed", 2),
    ("eed", 1),
    ("ing", 3),
    ("edly", 2),
    ("eedly", 1),
    ("ingly", 2),
];
const A6: Among = &[
    ("", 3),
    ("bb", 2),
    ("dd", 2),
    ("ff", 2),
    ("gg", 2),
    ("bl", 1),
    ("mm", 2),
    ("nn", 2),
    ("pp", 2),
    ("rr", 2),
    ("at", 1),
    ("tt", 2),
    ("iz", 1),
];
const A7: Among = &[
    ("anci", 3),
    ("enci", 2),
    ("ogi", 14),
    ("li", 16),
    ("bli", 12),
    ("abli", 4),
    ("alli", 8),
    ("fulli", 9),
    ("lessli", 15),
    ("ousli", 10),
    ("entli", 5),
    ("aliti", 8),
    ("biliti", 12),
    ("iviti", 11),
    ("tional", 1),
    ("ational", 7),
    ("alism", 8),
    ("ation", 7),
    ("ization", 6),
    ("izer", 6),
    ("ator", 7),
    ("iveness", 11),
    ("fulness", 9),
    ("ousness", 10),
    ("ogist", 13),
];
const A8: Among = &[
    ("icate", 4),
    ("ative", 6),
    ("alize", 3),
    ("iciti", 4),
    ("ical", 4),
    ("tional", 1),
    ("ational", 2),
    ("ful", 5),
    ("ness", 5),
];
const A9: Among = &[
    ("ic", 1),
    ("ance", 1),
    ("ence", 1),
    ("able", 1),
    ("ible", 1),
    ("ate", 1),
    ("ive", 1),
    ("ize", 1),
    ("iti", 1),
    ("al", 1),
    ("ism", 1),
    ("ion", 2),
    ("er", 1),
    ("ous", 1),
    ("ant", 1),
    ("ent", 1),
    ("ment", 1),
    ("ement", 1),
];
const A10: Among = &[("e", 1), ("l", 2)];
const A11: Among = &[
    ("andes", -1),
    ("atlas", -1),
    ("bias", -1),
    ("cosmos", -1),
    ("early", 6),
    ("gently", 4),
    ("howe", -1),
    ("idly", 3),
    ("news", -1),
    ("only", 7),
    ("singly", 8),
    ("skies", 2),
    ("skis", 1),
    ("sky", -1),
    ("ugly", 5),
];
const AS11: [&str; 8] = [
    "ski", "sky", "idl", "gentl", "ugli", "earli", "onli", "singl",
];

const V: &str = "aeiouy";
const V_WXY: &str = "Yaeiouwxy";
const VALID_LI: &str = "cdeghkmnrt";
const AEO: &str = "aeo";

struct Env {
    s: Vec<char>,
    cursor: usize,
    limit: usize,
    lb: usize,
    bra: usize,
    ket: usize,
    p1: usize,
    p2: usize,
    y_found: bool,
}

impl Env {
    fn at(&self, i: usize) -> char {
        self.s[i]
    }
    fn in_grouping(&mut self, g: &str) -> bool {
        if self.cursor >= self.limit || !g.contains(self.at(self.cursor)) {
            return false;
        }
        self.cursor += 1;
        true
    }
    fn go_in_grouping(&mut self, g: &str) -> bool {
        while self.cursor < self.limit {
            if !g.contains(self.at(self.cursor)) {
                return true;
            }
            self.cursor += 1;
        }
        false
    }
    fn go_out_grouping(&mut self, g: &str) -> bool {
        while self.cursor < self.limit {
            if g.contains(self.at(self.cursor)) {
                return true;
            }
            self.cursor += 1;
        }
        false
    }
    fn in_grouping_b(&mut self, g: &str) -> bool {
        if self.cursor <= self.lb || !g.contains(self.at(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }
    fn out_grouping_b(&mut self, g: &str) -> bool {
        if self.cursor <= self.lb || g.contains(self.at(self.cursor - 1)) {
            return false;
        }
        self.cursor -= 1;
        true
    }
    fn go_out_grouping_b(&mut self, g: &str) -> bool {
        while self.cursor > self.lb {
            if g.contains(self.at(self.cursor - 1)) {
                return true;
            }
            self.cursor -= 1;
        }
        false
    }
    fn eq_s_b(&mut self, w: &str) -> bool {
        let w: Vec<char> = w.chars().collect();
        if self.cursor < self.lb + w.len() || self.s[self.cursor - w.len()..self.cursor] != w[..] {
            return false;
        }
        self.cursor -= w.len();
        true
    }
    fn prev_is(&self, ch: char) -> bool {
        self.cursor > self.lb && self.at(self.cursor - 1) == ch
    }
    /// Longest entry ending at the cursor; moves the cursor before it. 0 = none.
    fn find_among_b(&mut self, table: Among) -> i32 {
        let mut best: Option<(usize, i32)> = None;
        for &(w, r) in table {
            let w: Vec<char> = w.chars().collect();
            let n = w.len();
            if self.cursor >= self.lb + n
                && self.s[self.cursor - n..self.cursor] == w[..]
                && best.is_none_or(|(b, _)| n > b)
            {
                best = Some((n, r));
            }
        }
        best.map_or(0, |(n, r)| {
            self.cursor -= n;
            r
        })
    }
    /// Longest entry starting at the cursor; moves the cursor past it. 0 = none.
    fn find_among(&mut self, table: Among) -> i32 {
        let mut best: Option<(usize, i32)> = None;
        for &(w, r) in table {
            let w: Vec<char> = w.chars().collect();
            let n = w.len();
            if self.cursor + n <= self.limit
                && self.s[self.cursor..self.cursor + n] == w[..]
                && best.is_none_or(|(b, _)| n > b)
            {
                best = Some((n, r));
            }
        }
        best.map_or(0, |(n, r)| {
            self.cursor += n;
            r
        })
    }
    fn slice_from(&mut self, w: &str) {
        let w: Vec<char> = w.chars().collect();
        let (bra, ket) = (self.bra, self.ket);
        let adjustment = w.len() as isize - (ket - bra) as isize;
        self.s.splice(bra..ket, w.iter().copied());
        self.limit = (self.limit as isize + adjustment) as usize;
        if self.cursor >= ket {
            self.cursor = (self.cursor as isize + adjustment) as usize;
        } else if self.cursor > bra {
            self.cursor = bra;
        }
        self.ket = bra + w.len();
    }
    fn slice_del(&mut self) {
        self.slice_from("");
    }
    fn r1(&self) -> bool {
        self.p1 <= self.cursor
    }
    fn r2(&self) -> bool {
        self.p2 <= self.cursor
    }
    /// Distance from the end, the way generated code saves a backward cursor.
    fn mark(&self) -> usize {
        self.limit - self.cursor
    }
    fn reset(&mut self, mark: usize) {
        self.cursor = self.limit - mark;
    }
}

fn prelude(e: &mut Env) {
    e.y_found = false;
    let v1 = e.cursor;
    e.bra = e.cursor;
    if e.cursor < e.limit && e.at(e.cursor) == '\'' {
        e.cursor += 1;
        e.ket = e.cursor;
        e.slice_del();
    }
    e.cursor = v1;
    e.bra = e.cursor;
    if e.cursor < e.limit && e.at(e.cursor) == 'y' {
        e.cursor += 1;
        e.ket = e.cursor;
        e.slice_from("Y");
        e.y_found = true;
    }
    e.cursor = v1;
    loop {
        let v4 = e.cursor;
        let found = loop {
            let v5 = e.cursor;
            if e.in_grouping(V) {
                e.bra = e.cursor;
                if e.cursor < e.limit && e.at(e.cursor) == 'y' {
                    e.cursor += 1;
                    e.ket = e.cursor;
                    e.cursor = v5;
                    break true;
                }
            }
            e.cursor = v5;
            if e.cursor >= e.limit {
                break false;
            }
            e.cursor += 1;
        };
        if !found {
            e.cursor = v4;
            break;
        }
        e.slice_from("Y");
        e.y_found = true;
    }
    e.cursor = v1;
}

/// R1 and R2. A word that ends before a region leaves it at the limit.
fn mark_regions(e: &mut Env) {
    e.p1 = e.limit;
    e.p2 = e.limit;
    let v1 = e.cursor;
    regions(e);
    e.cursor = v1;
}

fn regions(e: &mut Env) -> Option<()> {
    let v2 = e.cursor;
    if e.find_among(A0) == 0 {
        e.cursor = v2;
        e.go_out_grouping(V).then_some(())?;
        e.cursor += 1;
        e.go_in_grouping(V).then_some(())?;
        e.cursor += 1;
    }
    e.p1 = e.cursor;
    e.go_out_grouping(V).then_some(())?;
    e.cursor += 1;
    e.go_in_grouping(V).then_some(())?;
    e.cursor += 1;
    e.p2 = e.cursor;
    Some(())
}

fn shortv(e: &mut Env) -> bool {
    let save = e.cursor;
    if e.out_grouping_b(V_WXY) && e.in_grouping_b(V) && e.out_grouping_b(V) {
        return true;
    }
    e.cursor = save;
    if e.out_grouping_b(V) && e.in_grouping_b(V) && e.cursor <= e.lb {
        return true;
    }
    e.cursor = save;
    e.eq_s_b("past")
}

fn step_1a(e: &mut Env) -> bool {
    let v1 = e.mark();
    e.ket = e.cursor;
    if e.find_among_b(A1) == 0 {
        e.reset(v1);
    } else {
        e.bra = e.cursor;
        e.slice_del();
    }
    e.ket = e.cursor;
    let var = e.find_among_b(A2);
    if var == 0 {
        return false;
    }
    e.bra = e.cursor;
    match var {
        1 => e.slice_from("ss"),
        2 => {
            let v2 = e.mark();
            if e.cursor >= e.lb + 2 {
                e.cursor -= 2;
                e.slice_from("i");
            } else {
                e.reset(v2);
                e.slice_from("ie");
            }
        }
        3 => {
            if e.cursor <= e.lb {
                return false;
            }
            e.cursor -= 1;
            if !e.go_out_grouping_b(V) {
                return false;
            }
            e.cursor -= 1;
            e.slice_del();
        }
        _ => {}
    }
    true
}

/// The part of Step 1b that keeps a suffix; false sends it to the deletion path.
fn step_1b_keep(e: &mut Env, var: i32) -> bool {
    match var {
        1 => {
            let v2 = e.mark();
            if e.r1() {
                let v3 = e.mark();
                let exception = e.find_among_b(A3) != 0 && e.cursor <= e.lb;
                if !exception {
                    e.reset(v3);
                    e.slice_from("ee");
                }
            }
            e.reset(v2);
            true
        }
        2 => false,
        3 => match e.find_among_b(A4) {
            0 => false,
            1 => {
                let v4 = e.mark();
                if !e.out_grouping_b(V) || e.cursor > e.lb {
                    return false;
                }
                e.reset(v4);
                e.bra = e.cursor;
                e.slice_from("ie");
                true
            }
            _ => e.cursor <= e.lb,
        },
        _ => true,
    }
}

fn step_1b(e: &mut Env) -> bool {
    e.ket = e.cursor;
    let var = e.find_among_b(A5);
    e.bra = e.cursor;
    let v1 = e.mark();
    if step_1b_keep(e, var) {
        return true;
    }
    e.reset(v1);
    let v5 = e.mark();
    if !e.go_out_grouping_b(V) {
        return false;
    }
    e.reset(v5);
    e.slice_del();
    e.ket = e.cursor;
    e.bra = e.cursor;
    let v6 = e.mark();
    match e.find_among_b(A6) {
        1 => {
            e.slice_from("e");
            return false;
        }
        2 => {
            let v7 = e.mark();
            if e.in_grouping_b(AEO) && e.cursor <= e.lb {
                return false;
            }
            e.reset(v7);
        }
        _ => {
            if e.cursor != e.p1 {
                return false;
            }
            let v8 = e.mark();
            if !shortv(e) {
                return false;
            }
            e.reset(v8);
            e.slice_from("e");
            return false;
        }
    }
    e.reset(v6);
    e.ket = e.cursor;
    if e.cursor <= e.lb {
        return false;
    }
    e.cursor -= 1;
    e.bra = e.cursor;
    e.slice_del();
    true
}

fn step_1c(e: &mut Env) -> bool {
    e.ket = e.cursor;
    if e.prev_is('y') || e.prev_is('Y') {
        e.cursor -= 1;
    } else {
        return false;
    }
    e.bra = e.cursor;
    if !e.out_grouping_b(V) || e.cursor <= e.lb {
        return false;
    }
    e.slice_from("i");
    true
}

fn step_2(e: &mut Env) -> bool {
    e.ket = e.cursor;
    let var = e.find_among_b(A7);
    if var == 0 {
        return false;
    }
    e.bra = e.cursor;
    if !e.r1() {
        return false;
    }
    let to = match var {
        1 => "tion",
        2 => "ence",
        3 => "ance",
        4 => "able",
        5 => "ent",
        6 => "ize",
        7 => "ate",
        8 => "al",
        9 => "ful",
        10 => "ous",
        11 => "ive",
        12 => "ble",
        13 => "og",
        14 => {
            if !e.prev_is('l') {
                return false;
            }
            e.cursor -= 1;
            "og"
        }
        15 => "less",
        _ => {
            if !e.in_grouping_b(VALID_LI) {
                return false;
            }
            ""
        }
    };
    e.slice_from(to);
    true
}

fn step_3(e: &mut Env) -> bool {
    e.ket = e.cursor;
    let var = e.find_among_b(A8);
    if var == 0 {
        return false;
    }
    e.bra = e.cursor;
    if !e.r1() {
        return false;
    }
    match var {
        1 => e.slice_from("tion"),
        2 => e.slice_from("ate"),
        3 => e.slice_from("al"),
        4 => e.slice_from("ic"),
        5 => e.slice_del(),
        _ => {
            if !e.r2() {
                return false;
            }
            e.slice_del();
        }
    }
    true
}

fn step_4(e: &mut Env) -> bool {
    e.ket = e.cursor;
    let var = e.find_among_b(A9);
    if var == 0 {
        return false;
    }
    e.bra = e.cursor;
    if !e.r2() {
        return false;
    }
    if var != 1 {
        if e.prev_is('s') || e.prev_is('t') {
            e.cursor -= 1;
        } else {
            return false;
        }
    }
    e.slice_del();
    true
}

fn step_5(e: &mut Env) -> bool {
    e.ket = e.cursor;
    let var = e.find_among_b(A10);
    if var == 0 {
        return false;
    }
    e.bra = e.cursor;
    if var == 1 {
        if !e.r2() {
            if !e.r1() {
                return false;
            }
            let v1 = e.mark();
            if shortv(e) {
                return false;
            }
            e.reset(v1);
        }
    } else {
        if !e.r2() || !e.prev_is('l') {
            return false;
        }
        e.cursor -= 1;
    }
    e.slice_del();
    true
}

fn exception1(e: &mut Env) -> bool {
    e.bra = e.cursor;
    let var = e.find_among(A11);
    if var == 0 {
        return false;
    }
    e.ket = e.cursor;
    if e.cursor < e.limit {
        return false;
    }
    if var > 0 {
        e.slice_from(AS11[var as usize - 1]);
    }
    true
}

fn postlude(e: &mut Env) {
    if !e.y_found {
        return;
    }
    loop {
        let v1 = e.cursor;
        let found = loop {
            let v2 = e.cursor;
            e.bra = e.cursor;
            if e.cursor < e.limit && e.at(e.cursor) == 'Y' {
                e.cursor += 1;
                e.ket = e.cursor;
                e.cursor = v2;
                break true;
            }
            e.cursor = v2;
            if e.cursor >= e.limit {
                break false;
            }
            e.cursor += 1;
        };
        if !found {
            e.cursor = v1;
            break;
        }
        e.slice_from("y");
    }
}

pub fn stem(word: &str) -> String {
    let s: Vec<char> = word.chars().collect();
    let limit = s.len();
    let mut e = Env {
        s,
        cursor: 0,
        limit,
        lb: 0,
        bra: 0,
        ket: limit,
        p1: 0,
        p2: 0,
        y_found: false,
    };
    if exception1(&mut e) {
        return e.s.iter().collect();
    }
    e.cursor = 0;
    if e.limit < 3 {
        return e.s.iter().collect();
    }
    prelude(&mut e);
    mark_regions(&mut e);
    e.lb = e.cursor;
    e.cursor = e.limit;
    for step in [step_1a, step_1b, step_1c, step_2, step_3, step_4, step_5] {
        let m = e.mark();
        step(&mut e);
        e.reset(m);
    }
    e.cursor = e.lb;
    let v9 = e.cursor;
    postlude(&mut e);
    e.cursor = v9;
    e.s.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::stem;

    #[test]
    fn snowball_3_cases_that_snowball_2_got_differently() {
        for (word, want) in [
            ("added", "add"),
            ("organization", "organiz"),
            ("university", "universiti"),
            ("evening", "evening"),
            ("international", "internat"),
            ("pasted", "paste"),
            ("emergency", "emergenc"),
            ("graphs", "graph"),
            ("indexing", "index"),
            ("vocabulary", "vocabulari"),
            ("studies", "studi"),
            ("dying", "die"),
            ("skies", "sky"),
            ("early", "earli"),
            ("generously", "generous"),
        ] {
            assert_eq!(stem(word), want, "{word}");
        }
    }
}

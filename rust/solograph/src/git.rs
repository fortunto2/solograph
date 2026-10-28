//! Every git call goes through here, because two things went wrong when it did not.
//!
//! - **A hook's environment.** Inside a git hook, git exports GIT_DIR and
//!   GIT_INDEX_FILE, and they override `git -C`. A post-commit hook in a linked
//!   worktree that scanned the main checkout read the worktree's index instead: on
//!   27 Oct 2026 one Epiphan repo lost 15 files from the index for about 12 hours.
//! - **A git that never returns.** `log.showSignature` with a hanging gpg, or a
//!   stuck fsmonitor, held the nightly job forever with nothing in its log. The
//!   Python version had 5 s and 60 s timeouts; the port had dropped them.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What a hook exports and `-C` does not override.
const HOOK_ENV: [&str; 7] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
];

/// stdout of `git -C dir args…`, or None when it fails or outlives `timeout`.
pub fn run(dir: &Path, args: &[&str], timeout: Duration) -> Option<Vec<u8>> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(dir);
    // A signature check or an fsmonitor daemon is how a read-only call hangs.
    cmd.args([
        "-c",
        "log.showSignature=false",
        "-c",
        "core.fsmonitor=false",
    ]);
    cmd.args(args);
    for v in HOOK_ENV {
        cmd.env_remove(v);
    }
    output(cmd, timeout)
}

/// Run `cmd` with its stdout captured; kill it past `timeout`.
pub fn output(mut cmd: Command, timeout: Duration) -> Option<Vec<u8>> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Read on a thread so a large output cannot fill the pipe and stall the child.
    let mut out = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        out.read_to_end(&mut buf).map(|_| buf)
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if started.elapsed() < timeout => std::thread::sleep(Duration::from_millis(5)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let buf = reader.join().ok()?.ok()?;
    status.success().then_some(buf)
}

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use super::output;

    #[test]
    fn a_command_that_hangs_is_killed_at_the_timeout() {
        let mut cmd = Command::new("sleep");
        cmd.arg("5");
        let t = Instant::now();
        assert!(output(cmd, Duration::from_millis(200)).is_none());
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    }

    #[test]
    fn a_large_output_does_not_stall() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "head -c 3000000 /dev/zero"]);
        assert_eq!(
            output(cmd, Duration::from_secs(10)).map(|b| b.len()),
            Some(3_000_000)
        );
    }
}

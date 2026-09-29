# Set up the transcribe skill (instructions for an AI agent)

You are an AI coding agent (Claude Code, Codex or another agent that supports `SKILL.md` skills).
Your user wants to transcribe recordings on a running free-transcribe server. Follow these steps
on the user's machine, report briefly after each one, and talk to the user in their language.

The user's message should contain the server URL. If it does not, ask for it before step 3.

## 1. Install the skill

Pick the branch that matches you.

**Claude Code** — install it as a plugin so it updates itself:

```bash
CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1 claude plugin marketplace add vgmakeev/free-transcribe
claude plugin install free-transcribe@free-transcribe
```

Then turn on background updates: in `~/.claude/settings.json` set
`extraKnownMarketplaces["free-transcribe"].autoUpdate` to `true`. Merge into the existing file
with `jq` or `python3`; keep every other key as it is.

**Codex and other agents** — clone the repository once and link the skill into your personal
skills directory (`~/.agents/skills` for Codex; use your agent's equivalent otherwise):

```bash
REPO=~/.local/share/free-transcribe
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else git clone https://github.com/vgmakeev/free-transcribe.git "$REPO"; fi
mkdir -p ~/.agents/skills
ln -sfn "$REPO/plugins/free-transcribe/skills/transcribe" ~/.agents/skills/transcribe
```

Later updates are `git -C ~/.local/share/free-transcribe pull`.

## 2. Check ffmpeg

Run `command -v ffmpeg`. Without it the skill still works but uploads video files whole, which is
slow. If it is missing, offer to install it (`brew install ffmpeg` on macOS, the system package
manager on Linux) and do so only if the user agrees.

## 3. Store the server credentials

The server sits behind HTTP Basic auth. Ask the user for the login and password, or tell them
they can paste them as `login:password`. Do not repeat the password back and do not print it in
command output. Store two variables:

- `TRANSCRIBE_URL` — the server URL, without a trailing slash
- `TRANSCRIBE_AUTH` — `login:password`

(For direct access to the API without the proxy, `TRANSCRIBE_TOKEN` with the API token replaces
`TRANSCRIBE_AUTH`. Codex strips variables whose names contain `TOKEN` from command environments by
default, so prefer `TRANSCRIBE_AUTH` there.)

Where to store them:

- **Claude Code**: in the `env` object of `~/.claude/settings.json`, merged into the existing
  file. Claude Code passes these variables to every command it runs.
- **Codex and other agents**: as `export` lines in the user's shell startup file (`~/.zshrc` for
  zsh, `~/.bashrc` for bash). Wrap the values in single quotes; passwords often contain `$`. If
  the lines already exist, replace them instead of adding duplicates.

## 4. Verify

Run the script's check with the values passed explicitly, because the current session does not
see the new settings yet:

```bash
TRANSCRIBE_URL='…' TRANSCRIBE_AUTH='…' <skill dir>/scripts/transcribe.sh --check
```

`<skill dir>` is `~/.agents/skills/transcribe` for the clone install, or the path that
`claude plugin list` / the plugin cache under `~/.claude/plugins/cache/free-transcribe/` shows
for the Claude Code plugin. Expect `ok: … credentials accepted`. On `authentication rejected`
ask the user to recheck the login and password.

## 5. Finish

Tell the user:

- to restart the agent (and, for the shell startup file, open a new terminal) so it picks up the
  skill and the credentials;
- how to use it: "transcribe ~/Downloads/meeting.mp4, speakers are Anna and Ivan" — the
  transcript appears next to the recording as `meeting.transcript.md`;
- that in sandboxed agents (Codex) the first run asks to approve network access for the script.

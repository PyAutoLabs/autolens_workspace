# AI-Assisted Development

This project uses an AI-first development workflow. Most features, bug fixes, and improvements are implemented through AI coding agents (Claude Code, GitHub Copilot, OpenAI Codex) working from structured issue descriptions.

## How It Works

1. **Issues are the starting point** — Every task begins as a GitHub issue with a structured format: an overview, a human-readable plan, a detailed implementation plan (in a collapsible block), and optionally the original prompt that generated the issue.

2. **AI agents pick up issues** — Issues can be assigned to AI coding agents (e.g. GitHub Copilot) which read the issue description, `AGENTS.md`, and `CLAUDE.md` for context, then implement the changes autonomously.

3. **Human review** — All AI-generated pull requests are reviewed by maintainers before merging.

## Maintainer Workflow

Maintainer-driven dev work starts as a prompt file in
**[PyAutoPrompt](https://github.com/PyAutoLabs/PyAutoPrompt)** — the public
workflow repo that hosts the PyAuto task registry and the prompt-coupled
Claude Code skills. The pipeline:

1. Write the task as `PyAutoPrompt/<category>/<name>.md` (free-form markdown
   describing what to do, with `@RepoName/path/to/file.py` references).
2. `/start_dev <category>/<name>.md` — reads the prompt, audits the code,
   drafts the GitHub issue you see in this repo, and files it.
3. `/start_library` or `/start_workspace` — opens a feature worktree under
   `~/Code/PyAutoLabs-wt/<task-name>/`.
4. `/ship_library` / `/ship_workspace` — runs tests, opens the PR, and
   tracks state in `PyAutoPrompt/active.md`.

External contributors don't need PyAutoPrompt access — open an issue using
the templates in this repo and the same machinery handles it on our end.

## Creating an Issue

When opening an issue, please use the provided issue templates. The **Feature / Task Request** template follows our standard format:

- **Overview** — What and why, in 2-4 sentences
- **Plan** — High-level bullet points (human-readable)
- **Detailed implementation plan** — File paths, steps, key files (in a collapsible block)
- **Original Prompt** — If you used an AI to help draft the issue, include the original prompt

If your feature involves a specific calculation, algorithm, or small piece of functionality — **include example code**. Even a rough script, a working prototype, or a snippet showing the existing behaviour you want to change makes a huge difference. Code examples give AI agents and human contributors concrete context to work from, and dramatically reduce misunderstandings about what you're asking for.

This structure ensures that both human contributors and AI agents can understand and act on the issue effectively.

## Contributing Without AI

Traditional contributions are equally welcome! If you prefer to work without AI tools, simply follow the development setup and pull request guidelines below. The issue templates are helpful for any contributor, AI or human.

---

# Contributing

Contributions are welcome and greatly appreciated!

## Getting Started

1. Clone the repository
2. Create a feature branch:
    ```
    git checkout -b feature/name-of-your-branch
    ```

3. Only edit files in `scripts/`. Notebooks are auto-generated.

4. Test your changes:
    ```
    bash run_all_scripts.sh
    ```

5. Regenerate notebooks after scripts pass.

6. Commit, push, and submit a pull request.

### Pull Request Guidelines

1. Include a summary of which scripts were changed and why.
2. Confirm that notebooks were regenerated.
3. Preserve all docstrings, comments, and tutorial explanations.

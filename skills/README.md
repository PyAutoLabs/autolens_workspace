The `skills` folder contains Codex skills for working with **PyAutoLens** workspace outputs and workflows.

Skills are Markdown instruction files used by AI agents to load task-specific context. They are not part of the
**PyAutoLens** runtime API and are intended for users running Codex or another compatible AI agent from this workspace.

# Files

- `al_load_results`: Load a single completed lens model-fit from its output folder, including its `Tracer`, samples,
  model and FITS products, and route follow-up analysis to the relevant workspace guides.

# Using Skills With Codex

To make a skill available through Codex's dollar-sign skill API, expose it as a `SKILL.md` file under the Codex skills
directory, for example:

```bash
mkdir -p ~/.codex/skills/al_load_results
ln -sfn /path/to/autolens_workspace/skills/al_load_results.md ~/.codex/skills/al_load_results/SKILL.md
```

The skill can then be referenced as `$al_load_results`.

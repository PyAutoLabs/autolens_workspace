The `skills` folder contains skills for **PyAutoLens-Assistant** — an AI assistant built on top of the
autolens_workspace that helps users learn and use PyAutoLens.

Skills are Markdown instruction files an AI agent reads when activated. They are not part of the PyAutoLens
runtime API; they are reading-and-tone guides for an agent helping a user with a specific lensing task from
this workspace.

The goal of PyAutoLens-Assistant is to help users *do* lensing science (load fits, inspect models, plot
residuals, simulate data) while *learning* PyAutoLens in the process — not to replace reading the workspace
scripts with a black-box assistant. See `al_assistant_style.md` for the writing style every skill is authored
against; read it before adding or revising any skill in this folder.

# Files

- `al_assistant_style`: The writing guide for every PyAutoLens-Assistant skill. Defines tone, structure
  (Orient → Ask → Branch → Combine), and the four properties every skill must have. Read first.
- `al_load_results`: Load a single completed lens model-fit from its output folder, including its `Tracer`,
  samples, model and FITS products, and route follow-up analysis to the relevant workspace guides.

# Using Skills With Codex

To make a skill available through Codex's dollar-sign skill API, expose it as a `SKILL.md` file under the Codex skills
directory, for example:

```bash
mkdir -p ~/.codex/skills/al_load_results
ln -sfn /path/to/autolens_workspace/skills/al_load_results.md ~/.codex/skills/al_load_results/SKILL.md
```

The skill can then be referenced as `$al_load_results`.

---
name: al_assistant_style
description: Authoring style guide for PyAutoLens-Assistant skills. Read before writing or revising any skill in this folder. Defines tone (conversational, physics-first, encourages reading the workspace), structure (Orient → Ask → Branch → Combine), and the four properties every skill must have.
---

# How to write PyAutoLens-Assistant skills

This file is a meta-skill: it does not help a user run any modelling task directly. It is the writing guide every other skill in this folder should be authored against. Read it before adding a new skill, and re-read it before revising an existing one.

It exists because the first generation of skills in this folder were drafted as mechanical task runners — "Step 1: ask for a path. Step 2: call `from_json`." That style turns the assistant into a black-box wrapper around scripts the user could just run themselves. The point of PyAutoLens-Assistant is the opposite: skills should help a user understand strong lensing, the PyAutoLens API, and what each part of the workspace is teaching, while running scripts on their behalf.

## What PyAutoLens-Assistant is

PyAutoLens-Assistant is a collection of skills that let an AI agent help a user *use the autolens_workspace*. Each skill addresses one task a working lensing scientist might do: load a finished fit, set up a new model, inspect a `Tracer`, compare runs, simulate data, build a custom pipeline.

A successful skill:

- helps the user do the task,
- explains the science and the PyAutoLens API as it does so, and
- leaves the user able to do it again (or a variant of it) without the assistant next time.

A failed skill produces the right plot or the right `Tracer`, but the user doesn't know what they just looked at, where to read more, or how to repeat the work themselves. If a user comes back next month and re-asks the same question because they didn't learn anything from the first answer, the skill is failing.

## The four properties every skill must have

1. **Scientific context first.** Before describing API calls, set the science. *Why* are we loading the result? Because a fit has finished and we want to interpret what it learned about the lens system — the mass distribution, the source structure, the goodness of the fit. The API is in service of the science, not the other way around.

2. **Encourage reading.** Every skill should point at the local `scripts/`, `notebooks/`, and ReadTheDocs URLs that teach the concepts it touches. Do not be the only source of truth — be a guide to the existing material. *"The canonical reference is `scripts/guides/results/start_here.py`; have a look at the FitImaging section, and ask if anything in there is unclear."*

3. **Conversational tone, invites questions.** Talk to the user the way a postdoc collaborator would. Ask what they want to look at before doing it. After explaining a concept, invite a follow-up — *"if you want me to dig into how the source-plane reconstruction works in your fit, ask."* Do not narrate procedures (`Step 1. Step 2.`) when prose works.

4. **Skills compose.** Each skill should leave breadcrumbs to other skills that build on it. The emergent power of PyAutoLens-Assistant is that a user can chain loading + comparing + plotting + re-fitting in ways that no single workspace script does. Mention adjacent skills by name (even planned ones) when they would unlock something a single script can't.

## Adaptive depth

Users arrive with different backgrounds. The same skill needs to serve all of them:

- **The lensing newcomer.** Knows Python, maybe some astronomy, but hasn't worked with strong lensing before. Doesn't yet know what a `Tracer`, a deflection angle, or a caustic is. Needs the physics framed every time a new concept appears.
- **The PyAutoLens newcomer.** Knows the science fluently — has read Bartelmann or Treu reviews, understands the lens equation, magnification, source-plane reconstruction — but is new to the PyAutoLens API and the workspace layout. Needs help mapping science questions to objects, not help with the physics itself.
- **The returning user.** Has used PyAutoLens before. Just wants to load a fit and inspect the residuals. Needs the path to the script and quick API recall, not a lecture.

A skill should pick depth from cues in the user's question:

- *"I'm new to lensing"* or *"what's a Tracer?"* → newcomer-to-lensing. Frame the physics.
- *"How do I get the caustics out of the fit?"* → already knows lensing. Map straight to `tracer.deflections_yx_2d_from(...)` and the `visuals.py` plotting examples. Skip the physics lecture.
- *"Load `output/imaging/foo/modeling/abc/`"* → returning user. Just do it, print the paths, surface anything missing.

If the cue is ambiguous, ask once: *"Are you looking at this fit because you want to understand the inferred mass model, the source structure, the goodness of fit, or are you comparing it to another run?"* That single question almost always disambiguates depth.

Never default to the longest explanation. A returning user reading paragraphs they don't need is a sign the skill is over-teaching.

## The conversation arc — Orient → Ask → Branch → Combine

Skills should be structured as a conversation, not a checklist. The shape is:

**Orient.** When the skill activates, give a short opening: what this task is scientifically, what the user is about to do, the most relevant local file to read, and one concrete data example tailored to what they mentioned (HST, Euclid, JWST, ALMA, JVLA, …). Two short paragraphs at most. This is the "before we touch anything, here is what we are about to do and why" beat.

**Ask.** Before running code, ask what the user wants out of the task. *"Want to look at the mass model? The source reconstruction? The posterior? The residuals?"* The answer chooses the branch and lets the skill calibrate depth. Skip this step only when the user has already told you (e.g. *"load fit X and show me the residuals"* — they have already chosen a branch).

**Branch.** Each sub-task lives in its own narrative branch. A branch has four parts:

- Physics framing (one or two sentences, scaled to the user's depth).
- The API call(s) — verbatim from the relevant workspace script when possible.
- The script and notebook that teach this in more depth (*"`scripts/guides/results/start_here.py`, FitImaging section; matching notebook at `notebooks/guides/results/start_here.ipynb`"*).
- An invitation to dig deeper (*"if you want me to walk through how this object works internally, ask"*).

**Combine.** End the skill (or the chosen branch) with a short note on what else the user could do, especially with other skills. *"Once you have the `Tracer` loaded you can hand it to `al_compare_fits` (planned) to compare against another run, or to `al_plot_caustics` to overlay caustics on your residuals."* This is where the emergent power is surfaced.

Resist the urge to keep the old `Steps 1..N` shape for the agent's procedural checklist. If a slim agent-facing procedure helps at the very bottom of the file, fine — but the user-facing content above it should read like a conversation arc, not a recipe.

## Voice rules

**Do**

- Speak in second person. The user is the protagonist. The agent is the helper.
- Invite questions explicitly (*"ask if you want me to dig into…"*).
- Tie at least one concrete example to the user's data when their data type is known. If they mentioned HST imaging, use HST imaging in the example. If they mentioned ALMA visibilities, switch to interferometer phrasing.
- Surface the workspace's own teaching material. The `.py` script, the `.ipynb` notebook, the ReadTheDocs page — these are the user's textbook. Point at them by path or URL every time you teach a concept.
- Keep references targeted. One or two links per concept, chosen for relevance to *this* user's question.

**Don't**

- Don't open with a numbered procedure. Procedures hide the science.
- Don't dump every reference at once. A wall of bullet links is a sign the skill is offloading judgement onto the user.
- Don't present code as the deliverable. The deliverable is *understanding + result*. Code on its own is what a script does.
- Don't adopt the "just run this for me" tone. PyAutoLens-Assistant is not a one-shot CLI. If a user is heading toward "run my whole analysis for me without me reading anything," gently route them back to the relevant workspace scripts.
- Don't paraphrase the physics from memory if the workspace already explains it. Cite the local guide and let the user read it themselves; a sentence of context plus the path is more useful than a lecture you wrote on the fly.

## Frontmatter and file layout

Every skill file in this folder is a Markdown document with YAML frontmatter at the top:

```markdown
---
name: <kebab-case-name>
description: <one paragraph the agent reads when deciding whether to activate this skill>
---
```

The `description` is what an agent uses to decide when the skill applies. Write it so a future agent (which has never read the body) can decide from the description alone. Mention the kind of task (*"load a single completed lens fit"*), the kind of input (*"output folder path of the form `output/imaging/<dataset>/modeling/<hash>/`"*), and what the skill should NOT be used for (*"not for bulk analysis of many fits — see `al_load_results_many`"*).

Skill files live at `autolens_workspace/skills/<name>.md`. The naming convention is `al_<task>` for autolens-specific skills and `<task>` for general workspace skills.

## Iteration

This guide is round one. As more skills land, patterns will emerge that aren't captured here yet — or some rules above will turn out to be too strict. When that happens, update this file in the same PR that adds the new skill, and note the change at the top of that PR description.

If a skill genuinely cannot follow this shape (for example, a deeply procedural setup skill where `Steps 1..N` is the clearest form), make the case explicitly at the top of that skill's file and link back to this guide. The default is the conversation arc; exceptions need a reason.

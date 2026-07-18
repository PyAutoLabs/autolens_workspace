---
name: generate_and_merge
description: Build Jupyter notebooks from scripts, commit, push, raise a PR to main, merge it, then return to main locally.
---

Build notebooks for the autolens_workspace, then open and merge a PR into `main`, and return to `main` locally.

## Steps

1. **Check git state**

   Run `git status` and `git branch` to confirm the working tree state and current branch. If there are uncommitted changes outside `notebooks/`, stop and tell the user.

2. **Create a working branch**

   Create and check out a new branch named `notebooks-update-<YYYY-MM-DD>` (use today's date). If the branch already exists, check it out.

3. **Generate notebooks and the workspace catalogue**

   Run from the workspace root:
   ```bash
   PYTHONPATH=../PyAutoHands/autobuild python3 ../PyAutoHands/autobuild/generate.py autolens
   ```
   This regenerates all notebooks in `notebooks/` from `scripts/`. It may take a few minutes.

   In the **same pass** it also regenerates the LLM-facing workspace catalogue from the script
   docstrings — `llms-full.txt` (the expanded companion to the curated `llms.txt`) and
   `workspace_index.json` (the machine-readable index). These come from
   [PyAutoHands](https://github.com/PyAutoLabs/PyAutoHands)'s `navigator.py` and cannot drift out
   of sync with the scripts, since they regenerate alongside the notebooks. Do **not** hand-edit
   either file, and never touch the curated `llms.txt` here.

4. **Commit the generated notebooks and catalogue**

   Stage all changes under `notebooks/`, any root-level `*.ipynb` files, and the generated
   catalogue files, then commit:
   ```bash
   git add notebooks/ start_here.ipynb llms-full.txt workspace_index.json
   git commit -m "Build notebooks from scripts"
   ```
   Stage `llms-full.txt` and `workspace_index.json` by name (as above) — never `git add .`, so
   unrelated `dataset/` churn is not swept in. Do not stage the curated `llms.txt`. If there is
   nothing to commit (notebooks and catalogue already up to date), tell the user and stop.

5. **Push the branch**

   ```bash
   git push -u origin <branch-name>
   ```

6. **Open a PR into `main`**

   Use the `gh` CLI to create a pull request targeting `main` (NOT `release`):
   ```bash
   gh pr create --base main --title "Update notebooks" --body "Regenerated notebooks from scripts using generate.py."
   ```

7. **Merge the PR**

   Merge with a merge commit (not squash, not rebase) and delete the remote branch after:
   ```bash
   gh pr merge --merge --delete-branch
   ```

8. **Return to main locally**

   Check out `main` and pull the merged changes so the local branch is up to date:
   ```bash
   git checkout main
   git pull origin main
   ```

9. **Confirm success**

   Run `git log --oneline -5` to confirm the merge commit is present on `main`. Report the PR URL and merge commit to the user.

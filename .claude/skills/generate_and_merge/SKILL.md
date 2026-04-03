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

3. **Generate notebooks**

   Run from the workspace root:
   ```bash
   PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py autolens
   ```
   This regenerates all notebooks in `notebooks/` from `scripts/`. It may take a few minutes.

4. **Commit the generated notebooks**

   Stage all changes under `notebooks/` and any root-level `*.ipynb` files, then commit:
   ```bash
   git add notebooks/ start_here.ipynb
   git commit -m "Build notebooks from scripts"
   ```
   If there is nothing to commit (notebooks already up to date), tell the user and stop.

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

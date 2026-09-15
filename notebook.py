import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import subprocess
    import marimo as mo

    commit_message = mo.ui.text(value="Update notebook", label="Commit Message")
    push_btn = mo.ui.run_button(label="Commit & Push to Git")
    pull_btn = mo.ui.run_button(label="Pull from Git")

    mo.vstack([commit_message, mo.hstack([push_btn, pull_btn])])

    return commit_message, mo, os, pull_btn, push_btn, subprocess


@app.cell(hide_code=True)
def _(commit_message, mo, os, pull_btn, push_btn, subprocess):
    output = ""
    work_dir = "/marimo"
    gh_token = os.environ.get("GITHUB_TOKEN")
    repo_slug = "etienneunai/CITS4012-Project"


    def run_git(cmd, cwd=work_dir):
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        combined = (res.stdout + "\n" + res.stderr).strip()
        # Mask token from error messages to prevent accidental leaks
        if gh_token:
            combined = combined.replace(gh_token, "[REDACTED_TOKEN]")
        return res.returncode, combined


    if push_btn.value:
        if not gh_token:
            output = "Error: GITHUB_TOKEN environment variable is not set."
        else:
            remote_url = f"https://oauth2:{gh_token}@github.com/{repo_slug}.git"
            logs = []

            # 1. Global configs executed from /root to bypass container permission checks
            run_git(
                ["git", "config", "--global", "--add", "safe.directory", work_dir],
                cwd="/root",
            )
            run_git(
                ["git", "config", "--global", "user.name", "Etienne Vinton Horn"],
                cwd="/root",
            )
            run_git(
                [
                    "git",
                    "config",
                    "--global",
                    "user.email",
                    "etienneunai@gmail.com",
                ],
                cwd="/root",
            )
            run_git(
                ["git", "config", "--global", "pull.rebase", "false"], cwd="/root"
            )

            # 2. Auto-initialize repository if runtime was restarted
            if not os.path.exists(os.path.join(work_dir, ".git")):
                run_git(["git", "init"])
                run_git(["git", "remote", "add", "origin", remote_url])

            # 3. Ensure branch is main (resolves "src refspec main does not match any")
            run_git(["git", "branch", "-M", "main"])

            # 4. Stage and commit
            run_git(["git", "add", "notebook.py"])
            code, commit_out = run_git(
                ["git", "commit", "-m", commit_message.value or "Update notebook"]
            )
            if code != 0 and "nothing to commit" not in commit_out:
                logs.append(f"Commit status:\n{commit_out}")

            # 5. Push current HEAD to remote main branch
            code, push_out = run_git(["git", "push", "-u", remote_url, "HEAD:main"])
            if code != 0:
                logs.append(f"Push failed:\n{push_out}")
            else:
                logs.append(push_out or "Pushed successfully to main.")

            output = "\n\n".join(logs)

    elif pull_btn.value:
        if not gh_token:
            output = "Error: GITHUB_TOKEN environment variable is not set."
        elif not os.path.exists(os.path.join(work_dir, ".git")):
            output = "Error: Repository not initialized. Click 'Commit & Push' once first."
        else:
            remote_url = f"https://oauth2:{gh_token}@github.com/{repo_slug}.git"
            run_git(
                ["git", "config", "--global", "--add", "safe.directory", work_dir],
                cwd="/root",
            )
            run_git(
                ["git", "config", "--global", "pull.rebase", "false"], cwd="/root"
            )

            # Reconcile divergent branch history with merge strategy
            code, pull_out = run_git(
                ["git", "pull", "--no-rebase", remote_url, "main"]
            )
            if "unrelated histories" in pull_out:
                code, pull_out = run_git(
                    [
                        "git",
                        "pull",
                        "--no-rebase",
                        "--allow-unrelated-histories",
                        remote_url,
                        "main",
                    ]
                )

            output = pull_out

    mo.md(f"```text\n{output}\n```") if output else None
    return


if __name__ == "__main__":
    app.run()

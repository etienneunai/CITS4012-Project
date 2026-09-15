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


@app.cell
def _(commit_message, mo, os, pull_btn, push_btn, subprocess):
    output = ""
    work_dir = "/marimo"
    gh_token = os.environ.get("GITHUB_TOKEN")
    repo_slug = "etienneunai/CITS4012-Project"

    if push_btn.value:
        if not gh_token:
            output = "Error: GITHUB_TOKEN environment variable is not set."
        else:
            remote_url = f"https://oauth2:{gh_token}@github.com/{repo_slug}.git"
        
            # 1. Global Git configuration (works regardless of directory status)
            config_cmds = [
                ["git", "config", "--global", "--add", "safe.directory", work_dir],
                ["git", "config", "--global", "user.name", "MoLab User"],
                ["git", "config", "--global", "user.email", "molab@example.com"],
            ]
            for cmd in config_cmds:
                subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

            # 2. Initialize and link repo if .git does not exist
            init_needed = not os.path.exists(os.path.join(work_dir, ".git"))
            if init_needed:
                init_res = subprocess.run(["git", "init"], cwd=work_dir, capture_output=True, text=True)
                if init_res.returncode != 0:
                    output = f"Failed to initialize git:\n{init_res.stderr}"
                else:
                    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=work_dir, capture_output=True, text=True)

            if not output:
                # 3. Fetch remote to reconcile branch history if repository already exists on GitHub
                if init_needed:
                    subprocess.run(["git", "fetch", remote_url, "main"], cwd=work_dir, capture_output=True, text=True)
                    subprocess.run(["git", "branch", "-M", "main"], cwd=work_dir, capture_output=True, text=True)

                commands = [
                    ["git", "add", "notebook.py"],
                    ["git", "commit", "-m", commit_message.value or "Update notebook"],
                    ["git", "push", "-u", remote_url, "main"],
                ]
            
                results = []
                for cmd in commands:
                    res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
                    stdout = res.stdout.strip()
                    stderr = res.stderr.strip()

                    # Ignore clean tree exit codes on commit
                    if res.returncode != 0 and "nothing to commit" in (stdout + stderr):
                        results.append("Working tree clean (no new changes to commit).")
                        break
                    elif res.returncode != 0:
                        results.append(f"Failed: {' '.join(cmd)}\n{stderr or stdout}")
                        break
                
                    if stdout:
                        results.append(stdout)
                    elif stderr:
                        results.append(stderr)
                    
                output = "\n".join(results) or "Pushed successfully."

    elif pull_btn.value:
        if not os.path.exists(os.path.join(work_dir, ".git")):
            output = "Error: Not a git repository. Click 'Commit & Push' once first to initialize."
        else:
            remote_url = f"https://oauth2:{gh_token}@github.com/{repo_slug}.git"
            res = subprocess.run(["git", "pull", remote_url, "main"], cwd=work_dir, capture_output=True, text=True)
            output = res.stdout if res.returncode == 0 else res.stderr

    mo.md(f"```text\n{output}\n```") if output else None
    return


if __name__ == "__main__":
    app.run()

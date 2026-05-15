"""
Step 10 — GitHub + Cloudflare Pages Deployment
Pushes the dashboard to GitHub and deploys to Cloudflare Pages via Wrangler CLI.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, DASHBOARD, CF_PROJECT_NAME, GITHUB_REPO_NAME

GH_USER = os.getenv("GITHUB_USERNAME", "")


def run_cmd(cmd: list[str], cwd=None, check=True) -> subprocess.CompletedProcess:
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or BASE_DIR, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"      {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"      [!] stderr: {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def check_tools():
    missing = []
    for tool in ["git", "gh", "wrangler"]:
        r = subprocess.run(["which", tool], capture_output=True)
        if r.returncode != 0:
            missing.append(tool)
    if missing:
        print(f"[!] Missing tools: {', '.join(missing)}")
        print("    Install:")
        if "gh" in missing:
            print("    brew install gh && gh auth login")
        if "wrangler" in missing:
            print("    npm install -g wrangler && wrangler login")
        sys.exit(1)
    print("[✓] git, gh, wrangler all available")


def write_gitignore():
    gi = BASE_DIR / ".gitignore"
    gi.write_text("""# Python
__pycache__/
*.pyc
*.pkl
*.egg-info/
venv/

# Secrets — never commit these
.env
.kaggle/

# Large data files
datasets/
cleaned/
models/
""")


def write_cf_config():
    cfg = BASE_DIR / "wrangler.toml"
    cfg.write_text(f"""name = "{CF_PROJECT_NAME}"
compatibility_date = "2025-01-01"

[site]
bucket = "./dashboard"
""")


def github_push():
    print("\n[→] Setting up GitHub repository...")
    git_dir = BASE_DIR / ".git"

    if not git_dir.exists():
        run_cmd(["git", "init"])
        run_cmd(["git", "branch", "-M", "main"])

    write_gitignore()
    write_cf_config()

    # Stage only the dashboard + agent code (not large data files)
    run_cmd(["git", "add", "dashboard/", "agent/", ".gitignore", "wrangler.toml"])

    try:
        run_cmd(["git", "commit", "-m", "feat: real estate social media ML dashboard"])
    except RuntimeError:
        print("    [!] Nothing new to commit")

    # Create GitHub repo if it doesn't exist
    r = subprocess.run(["gh", "repo", "view", GITHUB_REPO_NAME],
                       capture_output=True, cwd=BASE_DIR)
    if r.returncode != 0:
        print(f"[→] Creating GitHub repo: {GITHUB_REPO_NAME}")
        run_cmd(["gh", "repo", "create", GITHUB_REPO_NAME,
                 "--public", "--source=.", "--remote=origin", "--push"])
    else:
        print(f"[→] Repo exists — pushing...")
        run_cmd(["git", "push", "-u", "origin", "main"], check=False)

    r2 = subprocess.run(["gh", "repo", "view", GITHUB_REPO_NAME, "--json", "url"],
                        capture_output=True, text=True, cwd=BASE_DIR)
    if r2.returncode == 0:
        info = json.loads(r2.stdout)
        print(f"[✓] GitHub: {info.get('url', '')}")
        return info.get("url", "")
    return ""


def cloudflare_deploy():
    print("\n[→] Deploying to Cloudflare Pages...")
    write_cf_config()

    # Create project if first time
    r = subprocess.run(
        ["wrangler", "pages", "project", "list"],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    if CF_PROJECT_NAME not in r.stdout:
        print(f"[→] Creating Cloudflare Pages project: {CF_PROJECT_NAME}")
        run_cmd(["wrangler", "pages", "project", "create", CF_PROJECT_NAME,
                 "--production-branch=main"], check=False)

    result = run_cmd(
        ["wrangler", "pages", "deploy", "dashboard",
         "--project-name", CF_PROJECT_NAME, "--branch", "main"],
        check=False,
    )

    url = ""
    for line in result.stdout.splitlines():
        if "pages.dev" in line or "https://" in line:
            url = line.strip()
            break

    if url:
        print(f"[✓] Cloudflare Pages: {url}")
    else:
        print("[!] Deploy may have succeeded — check Cloudflare dashboard")
    return url


def run():
    print("=" * 60)
    print("STEP 10 — GitHub + Cloudflare Deploy")
    print("=" * 60)

    if not (DASHBOARD / "index.html").exists():
        print("[!] Dashboard not built — run Step 09 first")
        sys.exit(1)

    check_tools()
    gh_url = github_push()
    cf_url = cloudflare_deploy()

    result = {"github_url": gh_url, "cloudflare_url": cf_url}
    summary = BASE_DIR / "DEPLOYMENT.json"
    with open(summary, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[✓] Step 10 complete")
    print(f"    GitHub   : {gh_url or '(see above)'}")
    print(f"    Cloudflare: {cf_url or '(see above)'}")
    return result


if __name__ == "__main__":
    run()

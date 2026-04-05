"""
Run tests/test_integration.py on Databricks serverless compute,
then run the full suite (excluding torch-dependent tests) if it passes.
"""

import os
import sys
import time
import base64
import pathlib

env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, SubmitTask
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/Shared/insurance-frequency-severity-test"
REPO_ROOT = pathlib.Path(__file__).parent


def upload_file(local_path: pathlib.Path, workspace_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(workspace_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=workspace_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )


def upload_tree(local_dir: pathlib.Path, workspace_base: str, extensions: set) -> int:
    count = 0
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and f.suffix in extensions and "__pycache__" not in str(f):
            rel = f.relative_to(local_dir.parent)
            ws_path = f"{workspace_base}/{rel}"
            upload_file(f, ws_path)
            count += 1
    return count


print("Uploading source and tests...")
n = upload_tree(REPO_ROOT / "src", WORKSPACE_DIR, {".py"})
n += upload_tree(REPO_ROOT / "tests", WORKSPACE_DIR, {".py"})
upload_file(REPO_ROOT / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")
print(f"  Uploaded {n} files")

NOTEBOOK_CONTENT = f"""\
# Databricks notebook source
import subprocess
import sys
import shutil
import os
import tempfile

subprocess.run(
    ["pip", "install", "--quiet",
     "scipy", "numpy", "pandas", "pyarrow", "pytest",
     "statsmodels", "matplotlib", "scikit-learn"],
    check=True, capture_output=True
)
print("Dependencies installed")

WORKSPACE_SRC = "{WORKSPACE_DIR}"
LOCAL_DIR = tempfile.mkdtemp(prefix="ins_freqsev_")
print(f"Working directory: {{LOCAL_DIR}}")

def copy_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for fname in files:
            if not fname.endswith('.pyc'):
                shutil.copy2(os.path.join(root, fname), os.path.join(dest_dir, fname))

copy_dir(WORKSPACE_SRC + "/src", LOCAL_DIR + "/src")
copy_dir(WORKSPACE_SRC + "/tests", LOCAL_DIR + "/tests")
shutil.copy2(WORKSPACE_SRC + "/pyproject.toml", LOCAL_DIR + "/pyproject.toml")
print(f"Copied project to {{LOCAL_DIR}}")

# Run integration tests first
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     LOCAL_DIR + "/tests/test_integration.py",
     "-v", "--tb=short", "--no-header",
     "-p", "no:cacheprovider",
     f"--rootdir={{LOCAL_DIR}}"],
    capture_output=True, text=True,
    cwd=LOCAL_DIR,
    env={{**os.environ, "PYTHONPATH": LOCAL_DIR + "/src"}}
)

output = result.stdout
if result.stderr:
    output += "\\nSTDERR:\\n" + result.stderr[:1000]
output += f"\\nExit code: {{result.returncode}}"
print(output[-12000:])

if result.returncode == 0:
    print("\\n=== Integration tests PASSED. Running full suite (no torch)... ===")
    result2 = subprocess.run(
        [sys.executable, "-m", "pytest",
         LOCAL_DIR + "/tests",
         "--ignore=" + LOCAL_DIR + "/tests/dependent",
         "-x", "-q", "--tb=short", "--no-header",
         "-p", "no:cacheprovider",
         f"--rootdir={{LOCAL_DIR}}"],
        capture_output=True, text=True,
        cwd=LOCAL_DIR,
        env={{**os.environ, "PYTHONPATH": LOCAL_DIR + "/src"}}
    )
    full_output = result2.stdout
    if result2.stderr:
        full_output += "\\nSTDERR:\\n" + result2.stderr[:500]
    full_output += f"\\nFull suite exit code: {{result2.returncode}}"
    print(full_output[-8000:])
    combined = output + "\\n\\n=== FULL SUITE (no torch/dependent) ===\\n" + full_output
    final_rc = result2.returncode
else:
    combined = output
    final_rc = result.returncode

shutil.rmtree(LOCAL_DIR, ignore_errors=True)

status = "PASSED" if final_rc == 0 else f"FAILED (exit {{final_rc}})"
dbutils.notebook.exit(f"{{status}}\\n{{combined[-10000:]}}")
"""

NOTEBOOK_PATH = f"{WORKSPACE_DIR}/_run_integration_tests"
content_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
try:
    w.workspace.mkdirs(path=WORKSPACE_DIR)
except Exception:
    pass
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=content_b64,
    overwrite=True,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
)
print(f"Uploaded test notebook to {NOTEBOOK_PATH}")

run_response = w.jobs.submit(
    run_name="insurance-freqsev-integration-pytest",
    tasks=[
        SubmitTask(
            task_key="pytest",
            notebook_task=NotebookTask(notebook_path=NOTEBOOK_PATH),
        )
    ],
)

run_id = run_response.run_id
print(f"\nSubmitted serverless run ID: {run_id}")

while True:
    state = w.jobs.get_run(run_id=run_id)
    life = str(state.state.life_cycle_state)
    result_state = str(state.state.result_state)
    print(f"  Status: {life} / {result_state}      ", end="\r")
    if any(x in life for x in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR")):
        print()
        break
    time.sleep(15)

for task in state.tasks or []:
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        print("\n" + "=" * 60)
        print("TEST OUTPUT:")
        print("=" * 60)
        if output.notebook_output and output.notebook_output.result:
            print(output.notebook_output.result)
        elif output.error:
            print("ERROR:", output.error)
            if output.error_trace:
                print("TRACE:", output.error_trace[:3000])
        else:
            print("(no output captured)")
    except Exception as e:
        print(f"Could not fetch output for task {task.task_key}: {e}")

final = str(state.state.result_state)
print(f"\nFinal result: {final}")
sys.exit(0 if "SUCCESS" in final else 1)

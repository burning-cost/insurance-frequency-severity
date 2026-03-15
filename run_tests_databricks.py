"""
Run insurance-frequency-severity tests on Databricks serverless.
"""
import base64
import os
import sys
import time

# Load credentials from env file
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as _f:
    for _line in _f:
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ[_k] = _v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    NotebookTask,
    RunLifeCycleState,
    RunResultState,
    SubmitTask,
)
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
HOST = "https://dbc-150a27f5-e1e7.cloud.databricks.com/"

notebook_content = """\
# Databricks notebook source
import subprocess
import sys
import shutil
import os

# Install dependencies
subprocess.run(
    ["pip", "install", "--quiet",
     "torch", "numpy", "pandas", "scipy", "statsmodels",
     "matplotlib", "scikit-learn", "pytest"],
    check=True, capture_output=True
)
print("Dependencies installed")

# Copy project to /tmp (Workspace is read-only for pycache)
src = "/Workspace/insurance-frequency-severity"
dst = "/tmp/insurance-frequency-severity"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
print(f"Copied to {dst}")

# Install the package from /tmp
subprocess.run(
    ["pip", "install", "--quiet", "-e", dst],
    check=True, capture_output=True
)
print("Package installed from /tmp")

# Run tests
r = subprocess.run(
    ["python", "-m", "pytest",
     f"{dst}/tests/",
     "--tb=short", "-v", "--no-header", "-p", "no:warnings",
     "-p", "no:cacheprovider"],
    capture_output=True, text=True,
    cwd=dst
)

full_output = r.stdout
if r.stderr:
    full_output += "\\nSTDERR:\\n" + r.stderr
print(full_output)

result_snippet = full_output[-5000:]
dbutils.notebook.exit(result_snippet)
"""

notebook_path = "/Workspace/Shared/insurance-frequency-severity-tests"
print("Uploading test notebook...")
encoded = base64.b64encode(notebook_content.encode()).decode()
w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True,
)
print("Upload OK")

print("Submitting (serverless)...")
run_response = w.jobs.submit(
    run_name="insurance-frequency-severity-pytest",
    tasks=[
        SubmitTask(
            task_key="pytest",
            notebook_task=NotebookTask(notebook_path=notebook_path),
        )
    ],
)

run_id = run_response.response.run_id
print(f"Run ID: {run_id}")
print(f"URL: {HOST}#job/runs/{run_id}")

print("\nPolling...")
while True:
    run = w.jobs.get_run(run_id=run_id)
    state = run.state
    lc = state.life_cycle_state
    print(f"  [{time.strftime('%H:%M:%S')}] {lc.value if lc else '?'}")
    if lc in (RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED, RunLifeCycleState.INTERNAL_ERROR):
        result_state = state.result_state
        print(f"  Result: {result_state.value if result_state else '?'}")
        print(f"  Msg: {state.state_message}")
        break
    time.sleep(30)

print("\n=== Task output ===")
try:
    tasks = run.tasks
    if tasks:
        task_run_id = tasks[0].run_id
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            print("NOTEBOOK RESULT:")
            print(output.notebook_output.result)
        if output.logs:
            print("LOGS:")
            print(output.logs[-6000:])
        if output.error:
            print("ERROR:", output.error)
        if output.error_trace:
            print("TRACE:", output.error_trace[-2000:])
except Exception as e:
    print(f"Output error: {e}")

success = (state.result_state == RunResultState.SUCCESS)
print(f"\n=== TEST {'PASSED' if success else 'FAILED'} ===")
sys.exit(0 if success else 1)

# Databricks notebook source
# MAGIC %pip install statsmodels matplotlib pandas numpy scipy pytest hatchling

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import subprocess, sys, os

# Add src to path
sys.path.insert(0, "/Workspace/insurance-frequency-severity/src")

import insurance_frequency_severity
print(f"Package version: {insurance_frequency_severity.__version__}")

# COMMAND ----------

import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-frequency-severity/tests/",
     "-v", "--tb=long", "--no-header",
     "--junit-xml=/tmp/test_results.xml"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-frequency-severity",
    env={**os.environ, "PYTHONPATH": "/Workspace/insurance-frequency-severity/src"}
)
output = result.stdout + "\nSTDERR:\n" + result.stderr
# Write to file for inspection
with open("/tmp/pytest_output.txt", "w") as f:
    f.write(output)
    f.write(f"\nReturn code: {result.returncode}\n")

# Print summary lines
lines = output.split("\n")
# Print last 100 lines (summary)
summary = "\n".join(lines[-100:])
print(summary)
print(f"\nReturn code: {result.returncode}")

# COMMAND ----------

# Print PASSED/FAILED counts
passed = output.count(" PASSED")
failed = output.count(" FAILED")
errors = output.count(" ERROR")
print(f"PASSED: {passed}, FAILED: {failed}, ERRORS: {errors}")

# Display full output for debugging
for i in range(0, len(output), 4000):
    print(output[i:i+4000])
    print("---")

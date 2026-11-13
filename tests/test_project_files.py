"""Checks on the non-code files: requirements, Dockerfile, CI, ignore files, README.

The Docker image itself cannot be built in the environment these tests were
written in, so these checks are the next best thing: they catch a COPY of a
file that does not exist, a requirement that does not parse, or a heavy import
that would make the small OCR image fail to start.
"""
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
HEAVY = {"torch", "transformers", "peft", "wandb", "bitsandbytes", "accelerate"}


def read(name):
    return (ROOT / name).read_text()


def requirement_lines(name):
    lines = []
    for raw in read(name).splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


# ---- requirements ----

@pytest.mark.parametrize(
    "name",
    ["requirements.txt", "requirements-dev.txt", "requirements-vlm.txt", "requirements-train.txt"],
)
def test_requirement_files_parse_and_includes_exist(name):
    lines = requirement_lines(name)
    assert lines, f"{name} is empty"
    for line in lines:
        if line.startswith("-r "):
            assert (ROOT / line[3:].strip()).is_file(), f"{name} includes a missing file: {line}"
        elif line.startswith("--"):
            continue
        else:
            Requirement(line)  # raises InvalidRequirement if malformed


def test_runtime_requirements_are_exactly_pinned():
    for line in requirement_lines("requirements.txt"):
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s]*", line), line


def test_runtime_requirements_cover_what_the_api_needs():
    names = {Requirement(line).name.lower() for line in requirement_lines("requirements.txt")}
    # fastapi brings pydantic and starlette. python-multipart is needed for uploads.
    assert {"fastapi", "uvicorn", "python-multipart", "pillow", "pytesseract"} <= names
    assert not names & HEAVY


def test_vlm_requirements_include_the_model_libraries():
    names = {Requirement(line).name.lower() for line in requirement_lines("requirements-vlm.txt")}
    assert {"torch", "transformers", "peft", "accelerate"} <= names


# ---- imports ----

def module_level_imports(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("folder", ["app", "receipts"])
def test_heavy_libraries_are_never_imported_at_module_level(folder):
    for path in sorted((ROOT / folder).glob("*.py")):
        assert not module_level_imports(path) & HEAVY, path.name


def test_app_imports_with_all_ml_libraries_blocked():
    code = (
        "import sys\n"
        f"for name in {sorted(HEAVY)!r}:\n"
        "    sys.modules[name] = None\n"
        "import app.main, receipts.predictors, receipts.training, receipts.vlm_io\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ---- Dockerfile ----

def dockerfile_lines():
    return [l.strip() for l in read("Dockerfile").splitlines() if l.strip() and not l.strip().startswith("#")]


def test_dockerfile_copies_only_paths_that_exist():
    copies = [l for l in dockerfile_lines() if l.upper().startswith("COPY ")]
    assert copies
    for line in copies:
        parts = [p for p in line.split()[1:] if not p.startswith("--")]
        for source in parts[:-1]:
            assert (ROOT / source).exists(), f"COPY of missing path: {source}"


def test_dockerfile_basics():
    text = read("Dockerfile")
    assert "tesseract-ocr" in text
    assert "app.main:app" in text
    assert "${PORT}" in text  # Cloud Run supplies the port this way
    assert re.search(r"^USER (?!root)\S+", text, re.M), "the container should not run as root"
    assert "requirements-vlm.txt" in text


def test_dockerfile_and_ci_use_the_same_python_minor_version():
    docker = re.search(r"^FROM python:(\d+\.\d+)", read("Dockerfile"), re.M).group(1)
    workflow = yaml.safe_load(read(".github/workflows/tests.yml"))
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith("actions/setup-python"):
                assert str(step["with"]["python-version"]) == docker


def test_dockerignore_keeps_private_data_out_but_not_the_code():
    ignored = {l.strip() for l in read(".dockerignore").splitlines() if l.strip()}
    assert {"data", "outputs", ".env"} <= ignored
    assert not ignored & {"app", "receipts", "requirements.txt", "requirements-vlm.txt"}


def test_gitignore_keeps_data_and_secrets_out():
    ignored = {l.strip() for l in read(".gitignore").splitlines() if l.strip() and not l.startswith("#")}
    assert {"data/", "outputs/", ".env"} <= ignored


# ---- CI ----

def test_ci_workflow_is_valid_and_runs_the_tests():
    workflow = yaml.safe_load(read(".github/workflows/tests.yml"))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare "on" as True
    assert "push" in triggers and "pull_request" in triggers
    assert set(workflow["jobs"]) == {"tests", "tests-with-model-code"}
    for name, job in workflow["jobs"].items():
        steps = job["steps"]
        uses = [s["uses"] for s in steps if "uses" in s]
        assert any(u.startswith("actions/checkout") for u in uses), name
        commands = "\n".join(s.get("run", "") for s in steps)
        assert "pytest" in commands, name
        assert "tesseract-ocr" in commands, name
        for included in re.findall(r"-r\s+(\S+)", commands):
            assert (ROOT / included).is_file(), f"{name} installs from a missing file: {included}"


# ---- README ----

def test_readme_documents_every_script():
    readme = read("README.md")
    for script in sorted((ROOT / "scripts").glob("*.py")):
        assert script.name in readme, f"README does not mention scripts/{script.name}"


def test_readme_exists_and_has_no_leftover_markers():
    text = read("README.md")
    assert len(text.split()) > 100
    assert "TODO" not in text.upper().replace("TO DO", ""), "README.md has a TODO marker"


# ---- what git must never take ----

@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_ignores_data_results_and_the_adapter_but_not_the_code(tmp_path):
    """Ask git itself, in a scratch repository that has only the project's .gitignore."""
    (tmp_path / ".gitignore").write_text(read(".gitignore"))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    def ignored(path):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=tmp_path)
        assert result.returncode in (0, 1), result
        return result.returncode == 0

    for path in (
        "data/test.jsonl",
        "data/images/001.jpg",
        "outputs/lora/adapter_model.safetensors",
        "reports/tuned.json",
        "models/lora/adapter_config.json",
        "models/lora/adapter_model.safetensors",
        "models/lora/run_config.json",
        "anything.safetensors",
        ".env",
    ):
        assert ignored(path), f"{path} would be committed"
    for path in ("models/README.md", "models/.gitkeep", "app/main.py", "Dockerfile", "deploy/model_card.md"):
        assert not ignored(path), f"{path} would be left out of the repository"


def test_gitattributes_keeps_line_endings_as_lf():
    assert "eol=lf" in read(".gitattributes")


def test_no_file_in_the_repository_holds_a_personal_email_address_or_phone_number():
    pattern = re.compile(r"[\w.+-]+@(?:gmail|outlook|hotmail|yahoo)\.com|\(\d{3}\) \d{3}-\d{4}")
    skipped = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
    for path in ROOT.rglob("*"):
        if path.is_file() and not skipped & set(path.relative_to(ROOT).parts):
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            assert not pattern.search(text), f"{path.relative_to(ROOT)} holds a personal address or number"

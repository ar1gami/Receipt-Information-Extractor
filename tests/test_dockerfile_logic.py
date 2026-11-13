"""Dry runs of the Dockerfile's logic, without Docker.

Docker could not be run where these tests were written, so the shell steps of
the Dockerfile are run here with stand-ins for pip, python, apt-get and the
like, and the ENV lines are expanded by a small copy of Docker's rules. This
checks the logic: which command runs when, in which order, what the image's
environment ends up as, and when the build must stop.

It does not check that the real commands work, such as downloading PyTorch.
Only a real docker build checks that.
"""
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell")


def parse_dockerfile():
    """A list of (INSTRUCTION, argument text). Continued lines are joined and comments are dropped."""
    instructions = []
    pending = ""
    for raw in (ROOT / "Dockerfile").read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or (not stripped and not pending):
            continue
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        text = (pending + line).strip()
        pending = ""
        keyword, _, rest = text.partition(" ")
        instructions.append((keyword.upper(), rest.strip()))
    assert not pending, "the Dockerfile ends inside a continued line"
    return instructions


_VAR = re.compile(r"\$(?:\{(?P<name>\w+)(?:(?P<op>:[-+])(?P<word>[^}]*))?\}|(?P<bare>\w+))")


def expand(value, lookup):
    """The forms of Docker variable expansion used here: $VAR, ${VAR}, ${VAR:-word}, ${VAR:+word}."""

    def replace(match):
        name = match.group("name") or match.group("bare")
        current = lookup.get(name, "")
        op = match.group("op")
        if op == ":-":
            return current if current else match.group("word")
        if op == ":+":
            return match.group("word") if current else ""
        return current

    return _VAR.sub(replace, value)


def final_env(build_args):
    """The ENV values the image ends up with. ARG defaults come from the file, build_args override them."""
    args, env = {}, {}
    for keyword, rest in parse_dockerfile():
        if keyword == "ARG":
            name, _, default = rest.partition("=")
            args[name] = build_args.get(name, default)
        elif keyword == "ENV":
            lookup = {**args, **env}  # every value in one ENV line is expanded against what came before it
            new = {}
            for token in shlex.split(rest):
                key, _, value = token.partition("=")
                new[key] = expand(value, lookup)
            env.update(new)
    return env


def run_steps():
    return [(i, rest) for i, (keyword, rest) in enumerate(parse_dockerfile()) if keyword == "RUN"]


def step_containing(text):
    found = [(i, body) for i, body in run_steps() if text in body]
    assert len(found) == 1, f"expected exactly one RUN step containing {text!r}, found {len(found)}"
    return found[0]


def index_of(keyword, containing=""):
    for i, (kw, rest) in enumerate(parse_dockerfile()):
        if kw == keyword and containing in rest:
            return i
    raise AssertionError(f"no {keyword} line containing {containing!r}")


# ---- the environment ----

@pytest.mark.parametrize(
    "variant, base, offline, model",
    [
        ("ocr", "", "", "HuggingFaceTB/SmolVLM-500M-Instruct"),
        ("vlm", "", "", "HuggingFaceTB/SmolVLM-500M-Instruct"),
        ("vlm", "org/other-model", "1", "org/other-model"),
    ],
)
def test_the_image_environment_follows_the_build_arguments(variant, base, offline, model):
    env = final_env({"VARIANT": variant, "BASE_MODEL": base})
    assert env["PREDICTOR"] == variant  # app/main.py accepts exactly "ocr" or "vlm"
    assert env["MODEL_ID"] == model
    assert env["HF_HUB_OFFLINE"] == offline  # only a baked image is told not to ask the hub
    assert env["HF_HOME"] == "/opt/hf"
    assert env["PORT"] == "8080"


def test_adapter_path_is_where_the_models_folder_is_copied_to():
    instructions = parse_dockerfile()
    workdir = [rest for kw, rest in instructions if kw == "WORKDIR"][-1]
    copies = [rest.split() for kw, rest in instructions if kw == "COPY" and rest.split()[0] == "models"]
    assert copies, "the models folder is not copied into the image"
    destination = copies[0][1]
    assert final_env({})["ADAPTER_PATH"] == f"{workdir}/{destination.lstrip('./')}/lora"


def test_app_settings_come_from_variables_the_app_really_reads():
    source = (ROOT / "app" / "main.py").read_text()
    env = final_env({"VARIANT": "vlm"})
    for name in ("PREDICTOR", "MODEL_ID", "ADAPTER_PATH"):
        assert name in env and f'"{name}"' in source


# ---- order of the steps ----

def test_steps_come_in_a_sensible_order():
    pip_step, _ = step_containing("pip install -r requirements.txt")
    bake_step, _ = step_containing("predictor_from_env")
    user_step, _ = step_containing("useradd")
    copy_code = index_of("COPY", "receipts")
    copy_app = index_of("COPY", "app")
    copy_models = index_of("COPY", "models")
    user = index_of("USER", "appuser")
    offline = index_of("ENV", "HF_HUB_OFFLINE")

    assert pip_step < copy_code, "install the requirements before copying the code, so the big layer is cached"
    assert max(copy_code, copy_app, copy_models) < bake_step, "the bake step imports the app and reads the adapter"
    assert user_step < user < bake_step, "the cache must be written by the user that runs the server"
    assert bake_step < offline, "going offline before the download would block the download"


# ---- the shell steps ----

@POSIX_ONLY
@pytest.mark.parametrize("index_and_body", run_steps(), ids=lambda pair: pair[1][:30])
def test_every_run_step_is_valid_shell(index_and_body):
    result = subprocess.run(["/bin/sh", "-n", "-c", index_and_body[1]], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


STUBBED = ("apt-get", "pip", "python", "useradd", "mkdir", "chown", "rm")


def run_with_stubs(body, tmp_path, env=None, fail=()):
    """Run a RUN step with every external command replaced by one that only writes its arguments to a log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "log.txt"
    log.write_text("")
    for name in STUBBED:
        script = bin_dir / name
        script.write_text(
            "#!/bin/sh\n"
            f'echo "{name} $*" >> "$STUB_LOG"\n'
            f'case ",$STUB_FAIL," in *,{name},*) exit 1;; esac\n'
            "exit 0\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    full_env = {"PATH": str(bin_dir), "STUB_LOG": str(log), "STUB_FAIL": ",".join(fail), **(env or {})}
    result = subprocess.run(["/bin/sh", "-c", body], env=full_env, capture_output=True, text=True)
    return result, [line for line in log.read_text().splitlines() if line]


@POSIX_ONLY
def test_system_packages_step_installs_tesseract_and_cleans_up(tmp_path):
    _, body = step_containing("apt-get install")
    result, log = run_with_stubs(body, tmp_path)
    assert result.returncode == 0
    assert log[0] == "apt-get update"
    assert "--no-install-recommends" in log[1] and log[1].endswith("tesseract-ocr")
    assert log[2].startswith("rm -rf /var/lib/apt/lists")


@POSIX_ONLY
def test_ocr_build_installs_only_the_small_requirements(tmp_path):
    _, body = step_containing("pip install -r requirements.txt")
    result, log = run_with_stubs(body, tmp_path, {"VARIANT": "ocr", "BASE_MODEL": ""})
    assert result.returncode == 0, result.stderr
    assert log == ["pip install -r requirements.txt"]


@POSIX_ONLY
def test_vlm_build_installs_cpu_pytorch_first_then_checks_it_is_not_a_cuda_build(tmp_path):
    _, body = step_containing("pip install -r requirements.txt")
    result, log = run_with_stubs(body, tmp_path, {"VARIANT": "vlm", "BASE_MODEL": ""})
    assert result.returncode == 0, result.stderr
    assert log[0] == "pip install -r requirements.txt"
    assert log[1] == "pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision"
    assert log[2] == "pip install -r requirements-vlm.txt"
    assert log[3].startswith("python -c") and "torch.version.cuda is None" in log[3]
    assert len(log) == 4


@POSIX_ONLY
def test_vlm_build_fails_when_the_cuda_check_fails(tmp_path):
    _, body = step_containing("pip install -r requirements.txt")
    result, _ = run_with_stubs(body, tmp_path, {"VARIANT": "vlm", "BASE_MODEL": ""}, fail=("python",))
    assert result.returncode != 0


@POSIX_ONLY
def test_a_failed_install_stops_the_chain(tmp_path):
    _, body = step_containing("pip install -r requirements.txt")
    result, log = run_with_stubs(body, tmp_path, {"VARIANT": "vlm", "BASE_MODEL": ""}, fail=("pip",))
    assert result.returncode != 0
    assert log == ["pip install -r requirements.txt"]  # nothing after the first failure ran


@POSIX_ONLY
@pytest.mark.parametrize(
    "variant, base, message",
    [("gpu", "", "VARIANT must be ocr or vlm"), ("ocr", "some/model", "BASE_MODEL needs VARIANT=vlm")],
)
def test_bad_build_arguments_fail_before_anything_is_installed(tmp_path, variant, base, message):
    _, body = step_containing("pip install -r requirements.txt")
    result, log = run_with_stubs(body, tmp_path, {"VARIANT": variant, "BASE_MODEL": base})
    assert result.returncode != 0
    assert message in result.stderr
    assert log == []


@POSIX_ONLY
def test_bake_step_does_nothing_without_a_base_model(tmp_path):
    _, body = step_containing("predictor_from_env")
    result, log = run_with_stubs(body, tmp_path, {"BASE_MODEL": ""})
    assert result.returncode == 0
    assert log == []


@POSIX_ONLY
def test_bake_step_loads_the_model_through_the_servers_own_start_up_function(tmp_path):
    _, body = step_containing("predictor_from_env")
    result, log = run_with_stubs(body, tmp_path, {"BASE_MODEL": "org/model"})
    assert result.returncode == 0
    assert len(log) == 1 and "from app.main import predictor_from_env" in log[0]


@POSIX_ONLY
def test_a_failing_bake_step_fails_the_build(tmp_path):
    _, body = step_containing("predictor_from_env")
    result, _ = run_with_stubs(body, tmp_path, {"BASE_MODEL": "org/model"}, fail=("python",))
    assert result.returncode != 0


# ---- the adapter from the hub ----

def test_adapter_download_comes_after_the_copy_and_before_the_user_and_the_bake():
    download, _ = step_containing("snapshot_download")
    bake, _ = step_containing("predictor_from_env")
    user_step, _ = step_containing("useradd")
    copy_models = index_of("COPY", "models")
    assert copy_models < download, "the copy of models/ would overwrite the download"
    assert download < user_step, "only root can write into the models folder that COPY made"
    assert download < bake, "the bake step loads the adapter"


@POSIX_ONLY
def test_without_adapter_repo_the_download_step_does_nothing(tmp_path):
    _, body = step_containing("snapshot_download")
    result, log = run_with_stubs(body, tmp_path, {"ADAPTER_REPO": "", "ADAPTER_REVISION": ""})
    assert result.returncode == 0
    assert log == []


@POSIX_ONLY
def test_adapter_repo_replaces_the_local_folder_with_the_download(tmp_path):
    _, body = step_containing("snapshot_download")
    adapter_path = final_env({})["ADAPTER_PATH"]
    env = {"ADAPTER_REPO": "someone/receipt-lora", "ADAPTER_REVISION": "", "ADAPTER_PATH": adapter_path}
    result, log = run_with_stubs(body, tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert len(log) == 2
    assert log[0] == f"rm -rf {adapter_path}"  # a stale local adapter must not mix with the downloaded one
    assert log[1].startswith("python -c ") and log[1].split()[-2:] == ["someone/receipt-lora", adapter_path]


@POSIX_ONLY
@pytest.mark.parametrize("failing", ["rm", "python"])
def test_a_failed_adapter_download_fails_the_build(tmp_path, failing):
    _, body = step_containing("snapshot_download")
    env = {"ADAPTER_REPO": "someone/receipt-lora", "ADAPTER_REVISION": "", "ADAPTER_PATH": "/srv/models/lora"}
    result, log = run_with_stubs(body, tmp_path, env, fail=(failing,))
    assert result.returncode != 0
    if failing == "rm":
        assert len(log) == 1  # the download is not attempted after a failed clean up


@POSIX_ONLY
def test_adapter_repo_without_the_vlm_variant_fails_before_anything_is_installed(tmp_path):
    _, body = step_containing("pip install -r requirements.txt")
    result, log = run_with_stubs(body, tmp_path, {"VARIANT": "ocr", "BASE_MODEL": "", "ADAPTER_REPO": "a/b"})
    assert result.returncode != 0
    assert "ADAPTER_REPO needs VARIANT=vlm" in result.stderr
    assert log == []


def _download_snippet():
    """The Python code of the download step, as it is written in the Dockerfile."""
    _, body = step_containing("snapshot_download")
    match = re.search(r'python -c "(?P<code>[^"]*)"', body)
    assert match, "could not find the python -c snippet"
    return match.group("code")


@pytest.mark.parametrize("revision, expected", [("", None), ("abc123", "abc123")])
def test_the_download_snippet_passes_the_repo_folder_and_revision_to_the_hub(tmp_path, revision, expected):
    """Run the real snippet against a fake huggingface_hub that records how it was called."""
    fake = tmp_path / "huggingface_hub"
    fake.mkdir()
    record = tmp_path / "call.json"
    (fake / "__init__.py").write_text(
        "import json, os\n"
        "def snapshot_download(repo_id, **kwargs):\n"
        f"    open({str(record)!r}, 'w').write(json.dumps([repo_id, kwargs]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", _download_snippet(), "someone/receipt-lora", "/srv/models/lora", revision],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    repo_id, kwargs = json.loads(record.read_text())
    assert repo_id == "someone/receipt-lora"
    assert kwargs == {"local_dir": "/srv/models/lora", "revision": expected}


def test_the_hub_function_the_download_snippet_calls_exists_when_the_hub_library_is_installed():
    hub = pytest.importorskip("huggingface_hub")
    import inspect

    parameters = inspect.signature(hub.snapshot_download).parameters
    assert {"repo_id", "local_dir", "revision"} <= set(parameters)


# ---- the baked code path really exists ----

def test_the_function_the_bake_step_calls_exists_and_needs_no_arguments():
    from app.main import predictor_from_env

    assert predictor_from_env.__code__.co_argcount == 0

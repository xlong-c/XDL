#!/usr/bin/env python3
"""离线依赖导出与 Python 环境打包工具。

本脚本支持两类操作，直接在 main() 里修改 task 和对应配置即可。

1. download-deps
   作用:
   - 根据 pyproject.toml 声明依赖，或当前环境的 pip freeze 结果，导出离线安装包。
   - 可选地把当前项目自身构建成 wheel, 便于目标机器直接离线安装。

   执行后文件夹变化:
   - 会创建 output 指定目录。
   - 会生成 requirements.txt、install_offline.sh、README.txt、manifest.json。
   - 会在 output/packages/ 中下载 wheel / sdist。
   - 如果 include_project_wheel=True, 会额外在 output/packages/ 中生成当前项目的 wheel。

   执行后环境变化:
   - 不会修改当前 Python 环境中已安装的包。
   - 只会调用 pip download / pip wheel 读取依赖并写出离线文件。
   - 如果 dry_run=True, 只生成说明文件和计划文件, 不真正下载包。

2. pack-env
   作用:
   - 直接把当前 virtualenv 或 conda 环境整体打成压缩包，便于复制到另一台机器。

   执行后文件夹变化:
   - 会在 output 指定位置生成环境压缩包。
   - 会额外生成一个 output.manifest.json, 用于记录本次打包信息。

   执行后环境变化:
   - 不会修改当前环境中的已安装包。
   - 只会读取当前环境目录并生成压缩包。
   - 如果 dry_run=True, 只打印计划命令, 不真正打包。

推荐场景说明:
- download-deps + mode="declared"
  适合项目依赖已经较完整地写在 pyproject.toml 中，希望导出的离线包更干净、更小，也更接近正式部署方式。

- download-deps + mode="freeze"
  适合当前环境里存在手动安装但未写入 pyproject.toml 的依赖，或者你更关注“当前机器能跑通的版本组合尽量原样迁移”。
  这种方式生成的 requirements 往往更长，也可能带上一些当前环境里的非项目包。

- pack-env
  适合环境里有较多编译型依赖、CUDA 依赖、私有包，或者你希望最快速地整体复制一套可运行环境，不想在目标机器重新解析和安装依赖。
  这种方式对目标机器要求更高，通常需要源机器和目标机器尽量保持一致，例如操作系统、CPU 架构、Python 主版本、glibc、CUDA 驱动栈等。

- 不确定选哪种时
  先用 download-deps + mode="declared"。
  如果目标机器安装后仍缺包，再改用 mode="freeze"。
  如果连 freeze 方式都难以稳定复现，再考虑 pack-env。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = __import__("tomli")


def load_requirement_type() -> Any:
    """延迟加载 Requirement，避免静态扫描把兼容回退误认为项目依赖。"""

    try:
        module = __import__("packaging.requirements", fromlist=["Requirement"])
    except ModuleNotFoundError:
        module = __import__("pip._vendor.packaging.requirements", fromlist=["Requirement"])
    return module.Requirement


Requirement = load_requirement_type()


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_PACKAGE_DIRNAME = "packages"


@dataclass
class ProjectMetadata:
    """项目元数据。"""

    name: str
    version: str
    dependencies: list[str]
    optional_dependencies: dict[str, list[str]]


@dataclass
class DownloadDepsConfig:
    """离线依赖下载配置。"""

    output: str
    mode: str = "declared"
    extras: list[str] = field(default_factory=list)
    additional_requirements: list[str] = field(default_factory=list)
    python_bin: str = sys.executable
    project_root: str = str(DEFAULT_PROJECT_ROOT)
    package_dirname: str = DEFAULT_PACKAGE_DIRNAME
    index_url: str | None = None
    extra_index_urls: list[str] = field(default_factory=list)
    find_links: list[str] = field(default_factory=list)
    platform: str | None = None
    python_version: str | None = None
    implementation: str | None = None
    abi: str | None = None
    include_project_wheel: bool = True
    dry_run: bool = True


@dataclass
class PackEnvConfig:
    """Python 环境打包配置。"""

    output: str
    pack_format: str = "auto"
    env_prefix: str | None = None
    dry_run: bool = True


class OfflineBundleError(RuntimeError):
    """脚本执行失败时抛出的统一异常。"""


def normalize_name(name: str) -> str:
    """按照 PEP 503 规则规整包名。"""

    return re.sub(r"[-_.]+", "-", name).lower()


def quote_command(cmd: Sequence[str]) -> str:
    """把命令数组转成可读字符串。"""

    return " ".join(shlex.quote(part) for part in cmd)


def log(message: str) -> None:
    """输出日志。"""

    print(f"[INFO] {message}")


def warn(message: str) -> None:
    """输出警告。"""

    print(f"[WARN] {message}")


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    dry_run: bool = False,
) -> str:
    """执行子命令。"""

    log(f"执行命令: {quote_command(cmd)}")
    if dry_run:
        return ""

    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=capture_output,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        details = stderr or stdout or "命令执行失败，但没有返回可读输出。"
        raise OfflineBundleError(details)

    if capture_output:
        return completed.stdout

    return ""


def ensure_directory(path: Path) -> None:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)


def load_project_metadata(pyproject_path: Path) -> ProjectMetadata:
    """从 pyproject.toml 读取项目依赖。"""

    with pyproject_path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)

    project = data.get("project")
    if not isinstance(project, dict):
        raise OfflineBundleError("pyproject.toml 缺少 [project] 配置。")

    name = project.get("name")
    version = project.get("version")
    dependencies = project.get("dependencies", [])
    optional_dependencies = project.get("optional-dependencies", {})

    if not isinstance(name, str) or not isinstance(version, str):
        raise OfflineBundleError("pyproject.toml 中的 project.name / project.version 非法。")
    if not isinstance(dependencies, list):
        raise OfflineBundleError("pyproject.toml 中的 project.dependencies 必须是数组。")
    if not isinstance(optional_dependencies, dict):
        raise OfflineBundleError(
            "pyproject.toml 中的 project.optional-dependencies 必须是表。"
        )

    extras: dict[str, list[str]] = {}
    for extra_name, requirements in optional_dependencies.items():
        if not isinstance(extra_name, str) or not isinstance(requirements, list):
            raise OfflineBundleError("存在非法的 optional-dependencies 配置。")
        extras[extra_name] = [str(item).strip() for item in requirements if str(item).strip()]

    return ProjectMetadata(
        name=name,
        version=version,
        dependencies=[str(item).strip() for item in dependencies if str(item).strip()],
        optional_dependencies=extras,
    )


def split_csv_values(values: Sequence[str]) -> list[str]:
    """拆分字符串列表里的逗号值。"""

    result: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def deduplicate(items: Iterable[str]) -> list[str]:
    """按原顺序去重。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_self_extra_requirement(spec: str, project_name: str) -> list[str] | None:
    """识别形如 xdl[cv,logging] 的自引用 extra。"""

    try:
        requirement = Requirement(spec)
    except Exception:
        return None

    if normalize_name(requirement.name) != normalize_name(project_name):
        return None

    return sorted(requirement.extras)


def expand_extra(
    extra_name: str,
    metadata: ProjectMetadata,
    *,
    resolved_extras: set[str],
    seen_stack: set[str],
) -> list[str]:
    """递归展开 extras，兼容 all -> xdl[cv] 这种自引用写法。"""

    if extra_name in resolved_extras:
        return []

    if extra_name in seen_stack:
        raise OfflineBundleError(f"检测到 extras 循环引用: {extra_name}")

    if extra_name not in metadata.optional_dependencies:
        available = ", ".join(sorted(metadata.optional_dependencies)) or "无"
        raise OfflineBundleError(
            f"extra `{extra_name}` 不存在。可用 extras: {available}"
        )

    seen_stack.add(extra_name)
    requirements: list[str] = []

    for spec in metadata.optional_dependencies[extra_name]:
        nested_extras = parse_self_extra_requirement(spec, metadata.name)
        if nested_extras:
            for nested_extra in nested_extras:
                requirements.extend(
                    expand_extra(
                        nested_extra,
                        metadata,
                        resolved_extras=resolved_extras,
                        seen_stack=seen_stack,
                    )
                )
            continue
        requirements.append(spec)

    seen_stack.remove(extra_name)
    resolved_extras.add(extra_name)
    return requirements


def resolve_declared_requirements(
    metadata: ProjectMetadata,
    selected_extras: Sequence[str],
    additional_requirements: Sequence[str],
) -> list[str]:
    """基于 pyproject 解析需要下载的 requirement 列表。"""

    requirements = list(metadata.dependencies)
    resolved_extras: set[str] = set()

    for extra_name in selected_extras:
        requirements.extend(
            expand_extra(
                extra_name,
                metadata,
                resolved_extras=resolved_extras,
                seen_stack=set(),
            )
        )

    requirements.extend(additional_requirements)
    return deduplicate(requirements)


def is_local_project_requirement(
    line: str,
    project_name: str,
    project_root: Path,
) -> bool:
    """判断 pip freeze 结果是否指向当前项目。"""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False

    normalized_project_name = normalize_name(project_name)
    resolved_root = str(project_root.resolve())

    if stripped.startswith("-e "):
        lowered = stripped.lower()
        if f"#egg={normalized_project_name}" in lowered:
            return True
        return resolved_root in stripped

    try:
        requirement = Requirement(stripped)
    except Exception:
        return False

    return normalize_name(requirement.name) == normalized_project_name


def resolve_freeze_requirements(
    python_bin: str,
    project_name: str,
    project_root: Path,
    additional_requirements: Sequence[str],
) -> list[str]:
    """基于当前环境 pip freeze 生成精确依赖。"""

    output = run_command(
        [python_bin, "-m", "pip", "freeze"],
        capture_output=True,
    )
    requirements: list[str] = []

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if is_local_project_requirement(stripped, project_name, project_root):
            continue
        requirements.append(stripped)

    requirements.extend(additional_requirements)
    return deduplicate(requirements)


def write_requirements(requirements_path: Path, requirements: Sequence[str]) -> None:
    """写出 requirements 文件。"""

    content = "\n".join(requirements).strip()
    requirements_path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def detect_python_version_label(python_bin: str) -> str:
    """读取指定解释器的主次版本号。"""

    try:
        output = run_command(
            [
                python_bin,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
        ).strip()
    except OfflineBundleError:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    return output or f"{sys.version_info.major}.{sys.version_info.minor}"


def build_download_command(
    *,
    python_bin: str,
    package_dir: Path,
    requirements_path: Path,
    index_url: str | None,
    extra_index_urls: Sequence[str],
    find_links: Sequence[str],
    platform: str | None,
    python_version: str | None,
    implementation: str | None,
    abi: str | None,
) -> list[str]:
    """构造 pip download 命令。"""

    command = [
        python_bin,
        "-m",
        "pip",
        "download",
        "--dest",
        str(package_dir),
        "-r",
        str(requirements_path),
    ]

    if index_url:
        command.extend(["--index-url", index_url])

    for extra_index_url in extra_index_urls:
        command.extend(["--extra-index-url", extra_index_url])

    for link in find_links:
        command.extend(["--find-links", link])

    target_selector_enabled = any([platform, python_version, implementation, abi])
    if platform:
        command.extend(["--platform", platform])
    if python_version:
        command.extend(["--python-version", python_version])
    if implementation:
        command.extend(["--implementation", implementation])
    if abi:
        command.extend(["--abi", abi])

    if target_selector_enabled:
        # pip 在跨平台解析依赖时要求明确只下载二进制包，否则会直接报错。
        command.extend(["--only-binary", ":all:"])

    return command


def build_project_wheel(
    *,
    python_bin: str,
    project_root: Path,
    package_dir: Path,
    dry_run: bool,
) -> None:
    """把当前项目自身构建成 wheel，便于目标机器离线安装。"""

    run_command(
        [
            python_bin,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(package_dir),
            str(project_root),
        ],
        cwd=project_root,
        dry_run=dry_run,
    )


def render_install_script(
    *,
    project_name: str,
    project_version: str,
    requirements_filename: str,
    package_dirname: str,
    include_project_wheel: bool,
) -> str:
    """生成离线安装脚本。"""

    lines = [
        "#!/usr/bin/env bash",
        "",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PACKAGE_DIR="${SCRIPT_DIR}/' + package_dirname + '"',
        'REQ_FILE="${SCRIPT_DIR}/' + requirements_filename + '"',
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        "",
        '"${PYTHON_BIN}" -m pip install --no-index --find-links "${PACKAGE_DIR}" -r "${REQ_FILE}"',
    ]

    if include_project_wheel:
        lines.append(
            '"${PYTHON_BIN}" -m pip install --no-index --find-links "${PACKAGE_DIR}" '
            f'"{project_name}=={project_version}"'
        )

    lines.append("")
    return "\n".join(lines)


def render_download_readme(
    *,
    metadata: ProjectMetadata,
    mode: str,
    extras: Sequence[str],
    requirements_filename: str,
    package_dirname: str,
    include_project_wheel: bool,
    target_platform: str | None,
    target_python_version: str | None,
    resolved_python_version: str,
    index_url: str | None,
    extra_index_urls: Sequence[str],
    additional_requirements: Sequence[str],
) -> str:
    """生成输出目录自说明文档。"""

    extras_text = ", ".join(extras) if extras else "无"
    extra_requirements_text = (
        ", ".join(additional_requirements) if additional_requirements else "无"
    )
    extra_index_text = ", ".join(extra_index_urls) if extra_index_urls else "无"
    project_wheel_text = "是" if include_project_wheel else "否"
    platform_text = target_platform or "与当前主机一致"
    python_text = target_python_version or resolved_python_version

    return f"""XDL 离线安装包说明
=====================

项目名: {metadata.name}
项目版本: {metadata.version}
导出模式: {mode}
extras: {extras_text}
额外 requirement: {extra_requirements_text}
包含项目 wheel: {project_wheel_text}
目标平台: {platform_text}
目标 Python 版本: {python_text}
index-url: {index_url or "默认 PyPI"}
extra-index-url: {extra_index_text}

目录结构
--------
- {requirements_filename}: 本次离线安装使用的 requirement 列表
- {package_dirname}/: 下载好的 wheel / sdist
- install_offline.sh: 目标主机上的安装脚本
- manifest.json: 导出元数据

目标机器安装步骤
----------------
1. 把整个输出目录复制到目标机器。
2. 在目标机器进入该目录。
3. 执行:

   bash install_offline.sh

注意事项
--------
- 如果没有显式指定目标平台参数，默认按当前机器的 Python 版本、平台和架构下载。
- PyTorch 常见的 CPU / CUDA wheel 往往需要额外索引，例如:
  --extra-index-url https://download.pytorch.org/whl/cu118
  或
  --extra-index-url https://download.pytorch.org/whl/cpu
- 如果你的运行环境里存在未写入 pyproject.toml 的依赖，优先使用 mode=freeze，
  或通过 --requirement 手动补充。
"""


def write_manifest(manifest_path: Path, data: dict[str, Any]) -> None:
    """写出 manifest.json。"""

    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_executable(path: Path) -> None:
    """给脚本增加可执行权限。"""

    current_mode = path.stat().st_mode
    path.chmod(current_mode | 0o111)


def detect_environment_prefix(explicit_prefix: str | None) -> Path:
    """检测当前可打包的 Python 环境目录。"""

    if explicit_prefix:
        env_prefix = Path(explicit_prefix).expanduser().resolve()
        if not env_prefix.exists():
            raise OfflineBundleError(f"指定的环境目录不存在: {env_prefix}")
        return env_prefix

    if os.environ.get("CONDA_PREFIX"):
        return Path(os.environ["CONDA_PREFIX"]).resolve()

    if os.environ.get("VIRTUAL_ENV"):
        return Path(os.environ["VIRTUAL_ENV"]).resolve()

    current_prefix = Path(sys.prefix).resolve()
    if (current_prefix / "pyvenv.cfg").exists():
        return current_prefix

    raise OfflineBundleError(
        "未检测到 virtualenv / conda 环境。可通过 --env-prefix 显式指定。"
    )


def resolve_env_packer(
    format_name: str,
    env_prefix: Path,
) -> str:
    """根据环境类型选择打包工具。"""

    if format_name == "conda-pack":
        return format_name

    if format_name == "venv-pack":
        return format_name

    if (env_prefix / "conda-meta").exists():
        return "conda-pack"

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).resolve() == env_prefix.resolve():
        return "conda-pack"

    return "venv-pack"


def ensure_tool_exists(tool_name: str) -> None:
    """确保系统中已安装指定命令。"""

    if shutil.which(tool_name):
        return

    if tool_name == "conda-pack":
        hint = "可先执行: pip install conda-pack"
    elif tool_name == "venv-pack":
        hint = "可先执行: pip install venv-pack"
    else:
        hint = "请先安装所需工具。"

    raise OfflineBundleError(f"未找到命令 `{tool_name}`。{hint}")


def handle_download_deps(config: DownloadDepsConfig) -> None:
    """处理 download-deps 子命令。"""

    project_root = Path(config.project_root).expanduser().resolve()
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        raise OfflineBundleError(f"未找到 pyproject.toml: {pyproject_path}")

    output_dir = Path(config.output).expanduser().resolve()
    package_dir = output_dir / config.package_dirname
    ensure_directory(output_dir)
    ensure_directory(package_dir)

    metadata = load_project_metadata(pyproject_path)
    selected_extras = split_csv_values(config.extras)
    additional_requirements = split_csv_values(config.additional_requirements)
    resolved_python_version = detect_python_version_label(config.python_bin)

    if config.mode == "declared":
        requirements = resolve_declared_requirements(
            metadata,
            selected_extras,
            additional_requirements,
        )
    elif config.mode == "freeze":
        requirements = resolve_freeze_requirements(
            config.python_bin,
            metadata.name,
            project_root,
            additional_requirements,
        )
    else:
        raise OfflineBundleError("download_config.mode 只支持 `declared` 或 `freeze`。")

    requirements_path = output_dir / "requirements.txt"
    write_requirements(requirements_path, requirements)

    download_command = build_download_command(
        python_bin=config.python_bin,
        package_dir=package_dir,
        requirements_path=requirements_path,
        index_url=config.index_url,
        extra_index_urls=config.extra_index_urls,
        find_links=config.find_links,
        platform=config.platform,
        python_version=config.python_version,
        implementation=config.implementation,
        abi=config.abi,
    )

    install_script_path = output_dir / "install_offline.sh"
    install_script_path.write_text(
        render_install_script(
            project_name=metadata.name,
            project_version=metadata.version,
            requirements_filename=requirements_path.name,
            package_dirname=config.package_dirname,
            include_project_wheel=config.include_project_wheel,
        ),
        encoding="utf-8",
    )
    ensure_executable(install_script_path)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        render_download_readme(
            metadata=metadata,
            mode=config.mode,
            extras=selected_extras,
            requirements_filename=requirements_path.name,
            package_dirname=config.package_dirname,
            include_project_wheel=config.include_project_wheel,
            target_platform=config.platform,
            target_python_version=config.python_version,
            resolved_python_version=resolved_python_version,
            index_url=config.index_url,
            extra_index_urls=config.extra_index_urls,
            additional_requirements=additional_requirements,
        ),
        encoding="utf-8",
    )

    manifest = {
        "command": "download-deps",
        "project": asdict(metadata),
        "mode": config.mode,
        "extras": selected_extras,
        "additional_requirements": additional_requirements,
        "python_bin": config.python_bin,
        "resolved_python_version": resolved_python_version,
        "target": {
            "platform": config.platform,
            "python_version": config.python_version,
            "implementation": config.implementation,
            "abi": config.abi,
        },
        "output_dir": str(output_dir),
        "package_dir": str(package_dir),
        "download_command": download_command,
        "include_project_wheel": config.include_project_wheel,
        "dry_run": config.dry_run,
    }
    write_manifest(output_dir / "manifest.json", manifest)

    if requirements:
        log(f"已解析 {len(requirements)} 条 requirement。")
    else:
        warn("requirements 为空，只会导出项目 wheel。")

    run_command(download_command, dry_run=config.dry_run)

    if config.include_project_wheel:
        build_project_wheel(
            python_bin=config.python_bin,
            project_root=project_root,
            package_dir=package_dir,
            dry_run=config.dry_run,
        )

    if config.dry_run:
        log(f"dry-run 完成，输出目录已生成: {output_dir}")
    else:
        log(f"离线依赖已导出到: {output_dir}")


def handle_pack_env(config: PackEnvConfig) -> None:
    """处理 pack-env 子命令。"""

    output_path = Path(config.output).expanduser().resolve()
    ensure_directory(output_path.parent)

    env_prefix = detect_environment_prefix(config.env_prefix)
    packer = resolve_env_packer(config.pack_format, env_prefix)
    if config.dry_run:
        if not shutil.which(packer):
            warn(f"未找到命令 `{packer}`，dry-run 仅输出计划，不执行实际打包。")
    else:
        ensure_tool_exists(packer)

    if packer == "conda-pack":
        command = [packer, "-p", str(env_prefix), "-o", str(output_path)]
    else:
        command = [packer, "-p", str(env_prefix), "-o", str(output_path)]

    run_command(command, dry_run=config.dry_run)

    manifest = {
        "command": "pack-env",
        "env_prefix": str(env_prefix),
        "packer": packer,
        "output": str(output_path),
        "dry_run": config.dry_run,
    }
    write_manifest(Path(f"{output_path}.manifest.json"), manifest)

    if config.dry_run:
        log(f"dry-run 完成，环境包将输出到: {output_path}")
    else:
        log(f"环境已打包到: {output_path}")


def main() -> int:
    """程序入口，直接在这里修改导出参数。"""

    task = "download-deps"

    download_config = DownloadDepsConfig(
        output=str(DEFAULT_PROJECT_ROOT / "offline_bundle"),
        mode="declared",
        extras=["all"],
        additional_requirements=["loguru"],
        python_bin=sys.executable,
        project_root=str(DEFAULT_PROJECT_ROOT),
        package_dirname=DEFAULT_PACKAGE_DIRNAME,
        index_url=None,
        extra_index_urls=[],
        find_links=[],
        platform=None,
        python_version=None,
        implementation=None,
        abi=None,
        include_project_wheel=True,
        dry_run=True,
    )

    pack_env_config = PackEnvConfig(
        output=str(DEFAULT_PROJECT_ROOT / "offline_bundle" / "xdl-env.tar.gz"),
        pack_format="auto",
        env_prefix=None,
        dry_run=True,
    )

    try:
        if task == "download-deps":
            handle_download_deps(download_config)
        elif task == "pack-env":
            handle_pack_env(pack_env_config)
        else:
            raise OfflineBundleError("task 只支持 `download-deps` 或 `pack-env`。")
    except OfflineBundleError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[ERROR] 用户中断执行。", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

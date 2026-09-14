import os
import sys
import shutil
from pathlib import Path

def find_system_python():
    # 1. sys.executable if running with system python
    if "venv" not in sys.executable.lower():
        return Path(sys.executable)

    # 2. Check python on PATH
    py_path = shutil.which("python")
    if py_path and "venv" not in py_path.lower():
        return Path(py_path)

    # 3. Check py launcher
    py3_path = shutil.which("py")
    if py3_path:
        return Path(py3_path)

    # 4. Check common Windows Python install locations
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        py_dir = Path(local_appdata) / "Programs" / "Python"
        if py_dir.exists():
            for p in py_dir.glob("Python*/python.exe"):
                if p.exists():
                    return p

    for root_dir in ["C:\\Program Files\\Python*", "C:\\Python*"]:
        for p in Path("C:\\").glob(root_dir):
            exe = p / "python.exe"
            if exe.exists():
                return exe

    return None

def fix_venv_cfg(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    cfg_file = base_dir / "venv" / "pyvenv.cfg"
    if not cfg_file.exists():
        print("[INFO] No existing venv/pyvenv.cfg found.")
        return False

    sys_py = find_system_python()
    if not sys_py:
        print("[WARN] Could not automatically locate system Python on this machine.")
        return False

    py_home = sys_py.parent
    print(f"[INFO] Current host Python detected at: {sys_py}")
    print(f"[INFO] Host Python home: {py_home}")

    try:
        lines = cfg_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith("home ="):
                new_lines.append(f"home = {py_home}")
            elif line.strip().startswith("executable ="):
                new_lines.append(f"executable = {sys_py}")
            elif line.strip().startswith("command ="):
                new_lines.append(f"command = {sys_py} -m venv {base_dir / 'venv'}")
            else:
                new_lines.append(line)

        cfg_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print("[SUCCESS] venv/pyvenv.cfg successfully updated for this machine!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update pyvenv.cfg: {e}")
        return False

if __name__ == "__main__":
    fix_venv_cfg()

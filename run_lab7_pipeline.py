
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Existing scripts from your Cacao Rating project
STAGES = [
    ("Data validation", SRC_DIR / "validate_data.py"),
    ("Preprocessing", SRC_DIR / "preprocess_pipeline.py"),
    ("Model training and MLflow tracking", SRC_DIR / "train_mlflow.py"),
    ("Output validation", SRC_DIR / "validate_outputs.py"),
]


def main():
    results = []
    overall_success = True

    print("=" * 60)
    print("LAB 7: CACAO RATING END-TO-END MLOPS PIPELINE")
    print("=" * 60)

    for stage_name, script_path in STAGES:
        print(f"\n[STAGE] {stage_name}")

        if not script_path.exists():
            message = f"Script not found: {script_path}"
            print(f"[ERROR] {message}")
            results.append({
                "stage": stage_name,
                "status": "FAILED",
                "message": message,
            })
            overall_success = False
            break

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        if result.stderr:
            print(result.stderr)

        status = "PASSED" if result.returncode == 0 else "FAILED"

        results.append({
            "stage": stage_name,
            "status": status,
            "return_code": result.returncode,
        })

        print(f"[{status}] {stage_name}")

        if result.returncode != 0:
            overall_success = False
            print("[STOP] Pipeline stopped because a stage failed.")
            break

    report = {
        "pipeline": "Cacao Rating MLOps Pipeline",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "PASSED" if overall_success else "FAILED",
        "stages": results,
    }

    report_path = REPORTS_DIR / "lab7_pipeline_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    log_path = LOGS_DIR / "lab7_pipeline.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"Pipeline status: {report['overall_status']}\n"
        )
        for item in results:
            log_file.write(
                f"{item['stage']}: {item['status']}\n"
            )

    print("\n" + "=" * 60)
    print(f"Pipeline status: {report['overall_status']}")
    print(f"Report saved: {report_path}")
    print(f"Log saved: {log_path}")
    print("=" * 60)

    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()

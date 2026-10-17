import subprocess
import sys


def execute_pipeline():

    print()
    print("==============================================")
    print("   LAB 5 - CACAO PRODUCTION DATA PIPELINE")
    print("==============================================")
    print()

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
    ]

    for script in scripts:

        print()
        print("----------------------------------------------")
        print(f"[INFO] Executing: {script}")
        print("----------------------------------------------")
        print()

        result = subprocess.run(
            [sys.executable, script]
        )

        if result.returncode != 0:

            print()
            print("[ERROR] Pipeline halted.")
            print(f"[ERROR] {script} failed.")

            sys.exit(1)

    print()
    print("==============================================")
    print("       LAB 5 PIPELINE COMPLETED")
    print("==============================================")
    print()

    print("[SUCCESS] Data validation completed.")
    print("[SUCCESS] Preprocessing completed.")
    print("[SUCCESS] Output validation completed.")
    print()


if __name__ == "__main__":
    execute_pipeline()
from __future__ import annotations

import argparse
import time
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-VL from ModelScope")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--output", default="/home/user/ljn/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1000)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.retries + 1):
        try:
            print(f"DOWNLOAD_ATTEMPT={attempt}/{args.retries}", flush=True)
            result = snapshot_download(
                model_id=args.model_id,
                local_dir=str(output),
                max_workers=args.workers,
            )
            print(f"MODEL_READY={result}", flush=True)
            return
        except Exception as error:
            print(f"DOWNLOAD_ERROR={error!r}", flush=True)
            if attempt == args.retries:
                raise
            time.sleep(15)


if __name__ == "__main__":
    main()

import json
import os
import threading
import time
from typing import Any


def slog(component: str, event: str, **fields: Any) -> None:
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "component": component,
        "event": event,
        **fields,
    }
    print(json.dumps(record, ensure_ascii=True, separators=(",", ":")))

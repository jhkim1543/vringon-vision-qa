# -*- coding: utf-8 -*-
"""server.py on port 5211 (secondary instance for parallel sessions)."""
import os
os.environ["VRINGON_QA_PORT"] = "5211"
import server  # noqa: F401  (imports app)
import uvicorn

if __name__ == "__main__":
    uvicorn.run(server.app, host="127.0.0.1", port=5211)

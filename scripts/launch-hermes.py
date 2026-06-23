#!/usr/bin/env python3
"""Launch hermes gateway with production env loaded from /opt/data/hermes/.env"""
import os, subprocess, sys

env_file = "/opt/data/hermes/.env"
env = dict(os.environ)

with open(env_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k:
                env[k] = v

env["HERMES_HOME"] = "/opt/data/hermes"

cmd = ["/workspace/.venvs/hermes-agent/bin/hermes", "gateway", "run"]
log = open("/opt/data/hermes/logs/gateway.log", "a")
proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
print(f"hermes gateway pid={proc.pid}")

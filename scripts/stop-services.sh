#!/usr/bin/env bash
# HerVid — stop Pipeline API + Orchestrator

echo "[hervid] stopping services..."
pkill -f "pipeline_api.py"   2>/dev/null && echo "  pipeline_api stopped" || echo "  pipeline_api was not running"
pkill -f "orchestrate.py serve" 2>/dev/null && echo "  orchestrator stopped" || echo "  orchestrator was not running"
rm -f /tmp/hervid-pipeline.pid /tmp/hervid-orchestrator.pid
echo "[hervid] done."

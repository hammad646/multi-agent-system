#!/usr/bin/env bash
# Creates the empty layout from SPEC.md section 4
set -e
mkdir -p data app/{agents,workflows,rag,db,scheduler,mcp_servers,testing,api} tests
for d in app app/agents app/workflows app/rag app/db app/scheduler app/mcp_servers app/testing app/api; do touch $d/__init__.py; done
for f in test_supervisor test_rag test_github test_calendar test_gmail test_scheduler test_approvals test_workflows; do touch tests/$f.py; done
echo "layout created"

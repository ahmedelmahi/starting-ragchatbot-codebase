#!/bin/bash

# Development quality check script
# Usage: ./scripts/check.sh [command]
# Commands:
#   format  - Format code with black
#   check   - Check formatting without making changes
#   test    - Run pytest
#   all     - Run all checks (default)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

format() {
    echo "Formatting code with black..."
    uv run black backend/
    echo "Formatting complete!"
}

check_format() {
    echo "Checking code formatting..."
    uv run black --check backend/
    echo "Format check passed!"
}

run_tests() {
    echo "Running tests..."
    cd backend && uv run pytest
    echo "Tests complete!"
}

run_all() {
    echo "Running all quality checks..."
    echo ""
    check_format
    echo ""
    run_tests
    echo ""
    echo "All checks passed!"
}

case "${1:-all}" in
    format)
        format
        ;;
    check)
        check_format
        ;;
    test)
        run_tests
        ;;
    all)
        run_all
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 [format|check|test|all]"
        exit 1
        ;;
esac

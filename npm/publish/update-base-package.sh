#!/usr/bin/env bash
set -euo pipefail

# Used in CI, extract here for readability

# Script to update version in base package.json
# Usage: update-base-package.sh <version>

VERSION="${1:?Missing version}"

echo "Updating base package.json to version $VERSION..."

# Find the package.json relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_JSON="$SCRIPT_DIR/../package.json"

if [[ ! -f "$PACKAGE_JSON" ]]; then
  echo "❌ Error: package.json not found at $PACKAGE_JSON"
  exit 1
fi

# Update version in base package.json
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$PACKAGE_JSON"

# Update optionalDependencies versions
sed -i.bak "s/\"codex-acp-darwin-arm64\": \".*\"/\"codex-acp-darwin-arm64\": \"$VERSION\"/" "$PACKAGE_JSON"
sed -i.bak "s/\"codex-acp-darwin-x64\": \".*\"/\"codex-acp-darwin-x64\": \"$VERSION\"/" "$PACKAGE_JSON"
sed -i.bak "s/\"codex-acp-linux-arm64\": \".*\"/\"codex-acp-linux-arm64\": \"$VERSION\"/" "$PACKAGE_JSON"
sed -i.bak "s/\"codex-acp-linux-x64\": \".*\"/\"codex-acp-linux-x64\": \"$VERSION\"/" "$PACKAGE_JSON"
sed -i.bak "s/\"codex-acp-win32-arm64\": \".*\"/\"codex-acp-win32-arm64\": \"$VERSION\"/" "$PACKAGE_JSON"
sed -i.bak "s/\"codex-acp-win32-x64\": \".*\"/\"codex-acp-win32-x64\": \"$VERSION\"/" "$PACKAGE_JSON"

# Remove backup file
rm -f "$PACKAGE_JSON.bak"

echo "✅ Updated package.json:"
cat "$PACKAGE_JSON"

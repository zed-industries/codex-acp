#!/usr/bin/env bash
set -euo pipefail

# Used in CI, extract here for readability

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "NPM Package Setup Validation"
echo "============================="
echo

check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo -e "${RED}✗ Required command not found: $1${NC}"
    exit 1
  fi
}

check_command node
check_command grep

# 1. Validate wrapper script syntax
echo "1. Validating wrapper script syntax..."
if node -c npm/bin/codex-acp.js 2>/dev/null; then
  echo -e "${GREEN}✓ Wrapper script syntax is valid${NC}"
else
  echo -e "${RED}✗ Wrapper script has syntax errors${NC}"
  exit 1
fi
echo

# 2. Validate package.json files
echo "2. Validating package.json files..."
if node -e "JSON.parse(require('fs').readFileSync('npm/package.json', 'utf8'))" 2>/dev/null; then
  echo -e "${GREEN}✓ Base package.json is valid${NC}"
else
  echo -e "${RED}✗ Base package.json is invalid${NC}"
  exit 1
fi

# 3. Check template has required placeholders
echo "3. Validating template placeholders..."
missing_placeholders=0
for placeholder in PACKAGE_NAME VERSION OS ARCH; do
  if ! grep -q "\${${placeholder}}" npm/template/package.json; then
    echo -e "${RED}✗ Template missing ${placeholder} placeholder${NC}"
    missing_placeholders=1
  fi
done

if [ $missing_placeholders -eq 0 ]; then
  echo -e "${GREEN}✓ Template has all required placeholders${NC}"
else
  exit 1
fi
echo

# 4. Check version consistency
echo "4. Checking version consistency..."
CARGO_VERSION=$(grep -m1 "^version" Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
NPM_VERSION=$(node -e "console.log(require('./npm/package.json').version)")

echo "   Cargo.toml version: $CARGO_VERSION"
echo "   npm package.json version: $NPM_VERSION"

if [ "$CARGO_VERSION" != "$NPM_VERSION" ]; then
  echo -e "${RED}✗ Version mismatch${NC}"
  exit 1
fi
echo -e "${GREEN}✓ Versions are in sync${NC}"
echo

# 5. Verify optional dependencies list
echo "5. Verifying platform packages..."
EXPECTED_PACKAGES=(
  "@zed-industries/codex-acp-darwin-arm64"
  "@zed-industries/codex-acp-darwin-x64"
  "@zed-industries/codex-acp-linux-arm64"
  "@zed-industries/codex-acp-linux-x64"
  "@zed-industries/codex-acp-windows-arm64"
  "@zed-industries/codex-acp-windows-x64"
)

missing_packages=0
for pkg in "${EXPECTED_PACKAGES[@]}"; do
  if ! grep -q "\"$pkg\":" npm/package.json; then
    echo -e "${RED}✗ Missing package: $pkg${NC}"
    missing_packages=1
  fi
done

if [ $missing_packages -eq 0 ]; then
  echo -e "${GREEN}✓ All platform packages listed in optionalDependencies${NC}"
else
  exit 1
fi
echo

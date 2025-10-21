#!/usr/bin/env bash
set -euo pipefail

# Used in CI, extract here for readability

# Script to create platform-specific npm packages from release artifacts
# Usage: create-platform-packages.sh <artifacts-dir> <output-dir> <version>

ARTIFACTS_DIR="${1:?Missing artifacts directory}"
OUTPUT_DIR="${2:?Missing output directory}"
VERSION="${3:?Missing version}"

echo "Creating platform-specific npm packages..."
echo "Artifacts: $ARTIFACTS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Version: $VERSION"
echo

mkdir -p "$OUTPUT_DIR"

# Define platform mappings: target -> (npm-os, npm-arch, binary-extension)
# Note: We only package gnu variants for Linux
declare -A platforms=(
  ["aarch64-apple-darwin"]="darwin arm64 "
  ["x86_64-apple-darwin"]="darwin x64 "
  ["x86_64-unknown-linux-gnu"]="linux x64 "
  ["aarch64-unknown-linux-gnu"]="linux arm64 "
  ["x86_64-pc-windows-msvc"]="windows x64 .exe"
  ["aarch64-pc-windows-msvc"]="windows arm64 .exe"
)

for target in "${!platforms[@]}"; do
  read os arch ext <<< "${platforms[$target]}"

  # Determine archive extension
  if [[ "$os" == "windows" ]]; then
    archive_ext="zip"
  else
    archive_ext="tar.gz"
  fi

  # Find and extract the archive
  archive_path=$(find "$ARTIFACTS_DIR" -name "*-${target}.${archive_ext}" | head -n 1)

  if [[ -z "$archive_path" ]]; then
    echo "⚠️  Warning: No archive found for target $target"
    continue
  fi

  echo "📦 Processing $target from $(basename "$archive_path")"

  # Create package name
  pkg_name="codex-acp-${os}-${arch}"
  pkg_dir="$OUTPUT_DIR/${pkg_name}"
  mkdir -p "${pkg_dir}/bin"

  # Extract binary
  if [[ "$archive_ext" == "zip" ]]; then
    unzip -q -j "$archive_path" "codex-acp${ext}" -d "${pkg_dir}/bin/"
  else
    tar xzf "$archive_path" -C "${pkg_dir}/bin/" "codex-acp${ext}"
  fi

  # Make binary executable (important for Unix-like systems)
  chmod +x "${pkg_dir}/bin/codex-acp${ext}" 2>/dev/null || echo "Failed to make binary executable"

  # Create package.json from template
  export PACKAGE_NAME="$pkg_name"
  export VERSION="$VERSION"
  export OS="$os"
  export ARCH="$arch"

  # Find the template relative to this script
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TEMPLATE_PATH="$SCRIPT_DIR/../template/package.json"

  envsubst < "$TEMPLATE_PATH" > "${pkg_dir}/package.json"

  echo "   ✓ Created package: ${pkg_name}"
done

echo
echo "✅ Platform packages created in: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR"

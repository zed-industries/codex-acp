#!/usr/bin/env bash
set -euo pipefail

: "${TARGET:?TARGET environment variable is required}"
: "${GITHUB_ENV:?GITHUB_ENV environment variable is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE environment variable is required}"

mapfile -t codex_tags < <(
  awk '
    /codex-/ && /https:\/\/github.com\/openai\/codex/ {
      if (match($0, /tag = "[^"]+"/)) {
        tag = substr($0, RSTART, RLENGTH)
        sub(/^tag = "/, "", tag)
        sub(/"$/, "", tag)
        print tag
      }
    }
  ' Cargo.toml | sort -u
)

if [[ "${#codex_tags[@]}" -ne 1 ]]; then
  echo "Expected one OpenAI Codex git tag, found: ${codex_tags[*]:-<none>}" >&2
  exit 1
fi
codex_tag="${codex_tags[0]}"

case "${TARGET}" in
  x86_64-unknown-linux-*)
    bwrap_target="x86_64-unknown-linux-musl"
    ;;
  aarch64-unknown-linux-*)
    bwrap_target="aarch64-unknown-linux-musl"
    ;;
  *)
    echo "Bundled bwrap is only available for Linux targets, got ${TARGET}" >&2
    exit 1
    ;;
esac

asset_name="bwrap-${bwrap_target}.tar.gz"
download_url="https://github.com/openai/codex/releases/download/${codex_tag}/${asset_name}"
release_dir="${GITHUB_WORKSPACE}/target/${TARGET}/release"
resources_dir="${release_dir}/codex-resources"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Downloading bundled bwrap from ${download_url}"
curl -fsSL "${download_url}" -o "${tmp_dir}/${asset_name}"
tar xzf "${tmp_dir}/${asset_name}" -C "${tmp_dir}" "bwrap-${bwrap_target}"

mkdir -p "${release_dir}" "${resources_dir}"
install -m 0755 "${tmp_dir}/bwrap-${bwrap_target}" "${release_dir}/bwrap"
install -m 0755 "${tmp_dir}/bwrap-${bwrap_target}" "${resources_dir}/bwrap"

digest="$(sha256sum "${tmp_dir}/bwrap-${bwrap_target}" | awk '{print $1}')"
echo "CODEX_BWRAP_SHA256=${digest}" >> "${GITHUB_ENV}"
echo "Installed upstream bwrap ${asset_name} with sha256:${digest}"

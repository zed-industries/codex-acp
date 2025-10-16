# NPM Publishing

This directory contains the setup for publishing `codex-acp` to npm as a native binary package.

## Architecture

The npm distribution follows the pattern described in [Packaging Rust for npm](https://blog.orhun.dev/packaging-rust-for-npm/):

1. **Platform-specific packages**: Each supported platform gets its own npm package containing the compiled binary
   - `codex-acp-darwin-arm64` (macOS Apple Silicon)
   - `codex-acp-darwin-x64` (macOS Intel)
   - `codex-acp-linux-arm64` (Linux ARM64)
   - `codex-acp-linux-x64` (Linux x64)
   - `codex-acp-windows-arm64` (Windows ARM64)
   - `codex-acp-windows-x64` (Windows x64)

2. **Base package** (`codex-acp`): The main package that:
   - Declares all platform packages as `optionalDependencies`
   - Provides a Node.js wrapper script that locates and executes the correct binary
   - Users install this package, and npm automatically installs the right platform package

### Manual Testing

To test the package structure locally:

```bash
# Build a release binary
cargo build --release

# Create a test package
mkdir -p test-npm/codex-acp-darwin-arm64/bin
cp target/release/codex-acp test-npm/codex-acp-darwin-arm64/bin/
export PACKAGE_NAME="codex-acp-darwin-arm64"
export VERSION="0.2.9"
export OS="darwin"
export ARCH="arm64"
envsubst < npm/template/package.json > test-npm/codex-acp-darwin-arm64/package.json

# Test the wrapper
cd npm
npm link ../test-npm/codex-acp-darwin-arm64
npm link
codex-acp --help
```

## Usage

After publishing, users can install and use the package:

```bash
# Install globally
npm install -g codex-acp

# Or run directly with npx
npx codex-acp

# Use in a project
npm install codex-acp
npx codex-acp
```

## Version Management

Version must be kept in sync between:

- `Cargo.toml` - Source of truth
- `npm/package.json` - Updated during release workflow

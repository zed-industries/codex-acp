#!/usr/bin/env node

const { spawnSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

// Map Node.js platform/arch to package names
function getPlatformPackage() {
  const platform = process.platform;
  const arch = process.arch;

  const platformMap = {
    darwin: {
      arm64: "codex-acp-darwin-arm64",
      x64: "codex-acp-darwin-x64",
    },
    linux: {
      arm64: "codex-acp-linux-arm64",
      x64: "codex-acp-linux-x64",
    },
    win32: {
      arm64: "codex-acp-windows-arm64",
      x64: "codex-acp-windows-x64",
    },
  };

  const packages = platformMap[platform];
  if (!packages) {
    console.error(`Unsupported platform: ${platform}`);
    process.exit(1);
  }

  const packageName = packages[arch];
  if (!packageName) {
    console.error(`Unsupported architecture: ${arch} on ${platform}`);
    process.exit(1);
  }

  return packageName;
}

// Locate the binary
function getBinaryPath() {
  const packageName = getPlatformPackage();
  const binaryName = process.platform === "win32" ? "codex-acp.exe" : "codex-acp";

  try {
    // Try to resolve the platform-specific package
    const packagePath = require.resolve(`${packageName}/package.json`);
    const binaryPath = path.join(path.dirname(packagePath), "bin", binaryName);

    if (existsSync(binaryPath)) {
      return binaryPath;
    }
  } catch (e) {
    // Package not found
  }

  console.error(
    `Failed to locate ${packageName} binary. This usually means the optional dependency was not installed.`
  );
  console.error(`Platform: ${process.platform}, Architecture: ${process.arch}`);
  process.exit(1);
}

// Execute the binary
function run() {
  const binaryPath = getBinaryPath();
  const result = spawnSync(binaryPath, process.argv.slice(2), {
    stdio: "inherit",
    windowsHide: true,
  });

  if (result.error) {
    console.error(`Failed to execute ${binaryPath}:`, result.error);
    process.exit(1);
  }

  process.exit(result.status || 0);
}

run();

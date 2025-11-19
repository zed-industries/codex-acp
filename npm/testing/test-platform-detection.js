#!/usr/bin/env node

/**
 * Test the platform detection logic from the wrapper script
 */

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
      arm64: "codex-acp-win32-arm64",
      x64: "codex-acp-win32-x64",
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

// Test all known platform/arch combinations
function testAllPlatforms() {
  const testCases = [
    { platform: "darwin", arch: "arm64", expected: "codex-acp-darwin-arm64" },
    { platform: "darwin", arch: "x64", expected: "codex-acp-darwin-x64" },
    { platform: "linux", arch: "arm64", expected: "codex-acp-linux-arm64" },
    { platform: "linux", arch: "x64", expected: "codex-acp-linux-x64" },
    { platform: "win32", arch: "arm64", expected: "codex-acp-win32-arm64" },
    { platform: "win32", arch: "x64", expected: "codex-acp-win32-x64" },
  ];

  console.log("Testing platform detection logic...\n");

  let allPassed = true;

  for (const testCase of testCases) {
    // Mock the platform and arch
    const originalPlatform = process.platform;
    const originalArch = process.arch;

    Object.defineProperty(process, "platform", {
      value: testCase.platform,
      configurable: true,
    });
    Object.defineProperty(process, "arch", {
      value: testCase.arch,
      configurable: true,
    });

    try {
      const result = getPlatformPackage();
      if (result === testCase.expected) {
        console.log(`✓ ${testCase.platform}-${testCase.arch} -> ${result}`);
      } else {
        console.error(
          `✗ ${testCase.platform}-${testCase.arch} -> Expected: ${testCase.expected}, Got: ${result}`,
        );
        allPassed = false;
      }
    } catch (e) {
      console.error(
        `✗ ${testCase.platform}-${testCase.arch} -> Error: ${e.message}`,
      );
      allPassed = false;
    } finally {
      // Restore original values
      Object.defineProperty(process, "platform", {
        value: originalPlatform,
        configurable: true,
      });
      Object.defineProperty(process, "arch", {
        value: originalArch,
        configurable: true,
      });
    }
  }

  console.log();
  if (allPassed) {
    console.log("✓ All platform detection tests passed!");
    return 0;
  } else {
    console.error("✗ Some platform detection tests failed");
    return 1;
  }
}

// Run tests
const exitCode = testAllPlatforms();

// Show current platform info
console.log("\nCurrent platform:");
console.log(`  Platform: ${process.platform}`);
console.log(`  Arch: ${process.arch}`);
console.log(`  Package: ${getPlatformPackage()}`);

process.exit(exitCode);

[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$BinaryPath
)

$ErrorActionPreference = 'Stop'

Write-Host "Windows code signing script for codex-acp"
Write-Host "Binary path: $BinaryPath"

# Verify the binary exists
if (-not (Test-Path $BinaryPath)) {
    Write-Error "Error: Binary not found at $BinaryPath"
    exit 1
}

$canCodeSign = $false

# Check if all required environment variables are present
$requiredVars = @(
    'AZURE_TENANT_ID',
    'AZURE_CLIENT_ID',
    'AZURE_CLIENT_SECRET',
    'ACCOUNT_NAME',
    'CERT_PROFILE_NAME',
    'ENDPOINT',
    'FILE_DIGEST',
    'TIMESTAMP_DIGEST',
    'TIMESTAMP_SERVER'
)

$missingVars = @()
foreach ($var in $requiredVars) {
    if (-not (Test-Path "env:$var") -or [string]::IsNullOrWhiteSpace((Get-Item "env:$var").Value)) {
        $missingVars += $var
    }
}

if ($missingVars.Count -eq 0) {
    $canCodeSign = $true
    Write-Host "All required environment variables found."
} else {
    Write-Host "Missing environment variables: $($missingVars -join ', ')"
}

if ($canCodeSign) {
    Write-Host "Signing binary with Azure Trusted Signing..."

    try {
        # Check if Az.CodeSigning module is available
        if (-not (Get-Module -ListAvailable -Name Az.CodeSigning)) {
            Write-Host "Installing Az.CodeSigning module..."
            Install-Module -Name Az.CodeSigning -Repository PSGallery -Force -Scope CurrentUser
        }

        # Import the module
        Import-Module Az.CodeSigning -ErrorAction Stop

        # Authenticate with Azure using service principal
        $securePassword = ConvertTo-SecureString $env:AZURE_CLIENT_SECRET -AsPlainText -Force
        $credential = New-Object System.Management.Automation.PSCredential($env:AZURE_CLIENT_ID, $securePassword)

        Connect-AzAccount -ServicePrincipal -Tenant $env:AZURE_TENANT_ID -Credential $credential -ErrorAction Stop | Out-Null

        Write-Host "Connected to Azure successfully."

        # Prepare signing parameters
        $params = @{
            Endpoint                  = $env:ENDPOINT
            CodeSigningAccountName    = $env:ACCOUNT_NAME
            CertificateProfileName    = $env:CERT_PROFILE_NAME
            FileDigest                = $env:FILE_DIGEST
            TimestampDigest           = $env:TIMESTAMP_DIGEST
            TimestampRfc3161          = $env:TIMESTAMP_SERVER
            Files                     = $BinaryPath
        }

        # Enable trace if requested
        if ($env:TRACE -and [System.Convert]::ToBoolean($env:TRACE)) {
            Set-PSDebug -Trace 2
        }

        # Invoke signing
        Write-Host "Invoking Trusted Signing..."
        Invoke-TrustedSigning @params

        Write-Host "✓ Successfully signed $BinaryPath"
        exit 0
    }
    catch {
        Write-Error "Failed to sign binary: $_"
        Write-Host "Error details: $($_.Exception.Message)"
        exit 1
    }
    finally {
        # Clean up
        if (Get-Command Disconnect-AzAccount -ErrorAction SilentlyContinue) {
            Disconnect-AzAccount -ErrorAction SilentlyContinue | Out-Null
        }
        Set-PSDebug -Off
    }
} else {
    exit 1
}

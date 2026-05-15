<#
.SYNOPSIS
    Run the full FUNSD experiment pipeline (three baselines + analysis).

.DESCRIPTION
    Sequentially trains OCR-only, Text+Layout, and LayoutLMv3 models,
    then runs unified evaluation & error analysis.  Each stage can be
    skipped individually via the -Skip* parameters.

.PARAMETER SkipTextOnly
    Skip the OCR-only baseline training.

.PARAMETER SkipTextLayout
    Skip the Text+Layout baseline training.

.PARAMETER SkipLayoutLMv3
    Skip the LayoutLMv3 baseline training.

.PARAMETER SkipAnalysis
    Skip the post-training analysis step.

.EXAMPLE
    # Run everything (from repo root, with docvlm conda env active)
    .\funsd_experiment\scripts\07_run_full_experiments.ps1

.EXAMPLE
    # Re-run only LayoutLMv3
    .\funsd_experiment\scripts\07_run_full_experiments.ps1 -SkipTextOnly -SkipTextLayout -SkipAnalysis
#>

param(
    [switch] $SkipTextOnly,
    [switch] $SkipTextLayout,
    [switch] $SkipLayoutLMv3,
    [switch] $SkipAnalysis
)

$ErrorActionPreference = "Stop"

# Resolve the repository root.
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName

# Create timestamped log file.
$LogDir  = Join-Path $RepoRoot "funsd_experiment\outputs\logs"
$Ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "full_experiment_${Ts}.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string] $Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Run-Stage {
    param(
        [string] $Name,
        [string[]] $Arguments,
        [switch] $Skip
    )
    if ($Skip) {
        Write-Log "[SKIP] $Name"
        return
    }
    Write-Log "============================================================"
    Write-Log "[START] $Name"
    Write-Log "[CMD]  $($Arguments -join ' ')"
    $start = Get-Date
    try {
        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = $Python
        $pinfo.RedirectStandardOutput = $true
        $pinfo.RedirectStandardError = $true
        $pinfo.UseShellExecute = $false
        $pinfo.CreateNoWindow = $true
        $pinfo.WorkingDirectory = $RepoRoot
        $pinfo.Arguments = $Arguments -join " "
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $pinfo
        $process.Start() | Out-Null
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($stdout) { $stdout -split "`n" | ForEach-Object { Write-Log "[OUT] $_" } }
        if ($stderr) { $stderr -split "`n" | ForEach-Object { Write-Log "[ERR] $_" } }
        $exitCode = $process.ExitCode
        $elapsed = ((Get-Date) - $start).TotalSeconds
        if ($exitCode -eq 0) {
            Write-Log "[PASS] $Name  (${elapsed}s)"
        } else {
            Write-Log "[FAIL] $Name  exit_code=$exitCode (${elapsed}s)"
            Write-Log "Experiment pipeline halted at stage: $Name"
            exit $exitCode
        }
    } catch {
        $elapsed = ((Get-Date) - $start).TotalSeconds
        Write-Log "[FAIL] $Name  (${elapsed}s)"
        Write-Log "[ERROR] $_"
        Write-Log "Experiment pipeline halted at stage: $Name"
        exit 1
    }
}

# ---- Main ----
Write-Log "========== FUNSD Full Experiment Pipeline =========="
Write-Log "Log file : $LogFile"
Write-Log "Repo root: $RepoRoot"

# Python resolution: prefer docvlm conda env, fall back to PATH.
$Python = "python"
if (Test-Path "E:\ai\anaconda\envs\docvlm\python.exe") {
    $Python = "E:\ai\anaconda\envs\docvlm\python.exe"
}
Write-Log "Python   : $Python"

# ----------------------------------------------------------
# Stage A: OCR-only baseline
# ----------------------------------------------------------
$ArgsA = @(
    "funsd_experiment/src/train/train_text_only.py",
    "--data_dir", "funsd_experiment/data/prepared",
    "--model_name", "distilbert-base-uncased",
    "--output_dir", "funsd_experiment/outputs/checkpoints/text_only",
    "--prediction_dir", "funsd_experiment/outputs/predictions",
    "--metric_dir", "funsd_experiment/outputs/metrics",
    "--report_table", "funsd_experiment/report/tables/text_only_metrics.md",
    "--max_length", "512",
    "--epochs", "5",
    "--batch_size", "4",
    "--learning_rate", "5e-5",
    "--seed", "42"
)
Run-Stage -Name "OCR-only baseline" -Arguments $ArgsA -Skip:$SkipTextOnly

# ----------------------------------------------------------
# Stage B: Text + Layout baseline
# ----------------------------------------------------------
$ArgsB = @(
    "funsd_experiment/src/train/train_text_layout.py",
    "--data_dir", "funsd_experiment/data/prepared",
    "--model_name", "distilbert-base-uncased",
    "--output_dir", "funsd_experiment/outputs/checkpoints/text_layout",
    "--prediction_dir", "funsd_experiment/outputs/predictions",
    "--metric_dir", "funsd_experiment/outputs/metrics",
    "--report_table", "funsd_experiment/report/tables/text_layout_metrics.md",
    "--max_length", "512",
    "--epochs", "5",
    "--batch_size", "4",
    "--learning_rate", "5e-5",
    "--bbox_hidden_size", "128",
    "--seed", "42"
)
Run-Stage -Name "Text+Layout baseline" -Arguments $ArgsB -Skip:$SkipTextLayout

# ----------------------------------------------------------
# Stage C: LayoutLMv3
# ----------------------------------------------------------
$ArgsC = @(
    "funsd_experiment/src/train/train_layoutlmv3.py",
    "--data_dir", "funsd_experiment/data/prepared",
    "--model_name", "microsoft/layoutlmv3-base",
    "--output_dir", "funsd_experiment/outputs/checkpoints/layoutlmv3",
    "--prediction_dir", "funsd_experiment/outputs/predictions",
    "--metric_dir", "funsd_experiment/outputs/metrics",
    "--report_table", "funsd_experiment/report/tables/layoutlmv3_metrics.md",
    "--max_length", "512",
    "--epochs", "5",
    "--batch_size", "2",
    "--learning_rate", "5e-5",
    "--seed", "42"
)
Run-Stage -Name "LayoutLMv3 baseline" -Arguments $ArgsC -Skip:$SkipLayoutLMv3

# ----------------------------------------------------------
# Stage D: Unified analysis
# ----------------------------------------------------------
$ArgsD = @(
    "funsd_experiment/scripts/06_analyze_results.py",
    "--metric_dir", "funsd_experiment/outputs/metrics",
    "--prediction_dir", "funsd_experiment/outputs/predictions",
    "--analysis_out", "funsd_experiment/outputs/analysis",
    "--report_table_dir", "funsd_experiment/report/tables",
    "--experiment_log", "funsd_experiment/report/experiment_log.md"
)
Run-Stage -Name "Unified analysis" -Arguments $ArgsD -Skip:$SkipAnalysis

# ----------------------------------------------------------
# Stage E: Update experiment log with formal metrics
# ----------------------------------------------------------
$ArgsE = @("funsd_experiment/scripts/07_update_experiment_log.py")
Run-Stage -Name "Update experiment log" -Arguments $ArgsE

Write-Log "============================================================"
Write-Log "[DONE] Full experiment pipeline completed."
Write-Log "Log file: $LogFile"

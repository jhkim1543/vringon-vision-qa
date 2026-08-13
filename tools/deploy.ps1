# Publish the VRINGON Vision QA demo to GitHub Pages (idempotent).
#
# The token lives in Windows Credential Manager; `git credential fill` can pop
# a GUI under the "manager" helper, so it is read directly with CredRead.
param(
  [string]$Repo  = "vringon-vision-qa",
  [string]$Owner = "jhkim1543",
  [string]$Desc  = "Footwear visual QA demo - part-conditioned anomaly detection, with an honest controlled-rig vs field-photo comparison"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$sig = @'
using System;
using System.Runtime.InteropServices;
public class CredR {
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct CREDENTIAL {
    public uint Flags; public uint Type; public IntPtr TargetName; public IntPtr Comment;
    public long LastWritten; public uint CredentialBlobSize; public IntPtr CredentialBlob;
    public uint Persist; public uint AttributeCount; public IntPtr Attributes;
    public IntPtr TargetAlias; public IntPtr UserName;
  }
  [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern bool CredRead(string target, uint type, uint flags, out IntPtr credential);
  [DllImport("advapi32.dll")] public static extern void CredFree(IntPtr cred);
  public static string Get(string target) {
    IntPtr p;
    if (!CredRead(target, 1, 0, out p)) return null;
    var c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
    string s = Marshal.PtrToStringUni(c.CredentialBlob, (int)(c.CredentialBlobSize/2));
    CredFree(p);
    return s;
  }
}
'@
if (-not ("CredR" -as [type])) { Add-Type -TypeDefinition $sig -Language CSharp }
$token = [CredR]::Get("git:https://github.com")
if (-not $token) { throw "GitHub token not found in Credential Manager" }
$hdr = @{ Authorization = "token $token"; "User-Agent" = "vringon-deploy"; Accept = "application/vnd.github+json" }

function Api($method, $path, $body) {
  $u = "https://api.github.com$path"
  if ($body) { Invoke-RestMethod -Method $method -Uri $u -Headers $hdr -Body ($body | ConvertTo-Json -Depth 6) -ContentType "application/json" }
  else { Invoke-RestMethod -Method $method -Uri $u -Headers $hdr }
}

# 1. repository (public, so Pages is available without a paid plan)
try {
  $r = Api GET "/repos/$Owner/$Repo" $null
  Write-Host "repo exists: $($r.full_name)"
} catch {
  $r = Api POST "/user/repos" @{ name = $Repo; description = $Desc; private = $false; has_issues = $false; has_wiki = $false }
  Write-Host "repo created: $($r.full_name)"
}

# 2. local git + push
Push-Location $root
# git writes advisories (CRLF, detached head, ...) to stderr, which PowerShell 5.1
# promotes to a terminating error; git's own exit codes are the real signal here.
$ErrorActionPreference = "Continue"
if (-not (Test-Path ".git")) { git init -q -b main }
git config user.name  "jhkim1543"
git config user.email "54016655+jhkim1543@users.noreply.github.com"
# PowerShell 5.1 turns a native command's stderr into a terminating error under
# ErrorActionPreference=Stop, so probe instead of removing blindly.
if ((git remote) -contains "origin") { git remote set-url origin "https://github.com/$Owner/$Repo.git" }
else { git remote add origin "https://github.com/$Owner/$Repo.git" }
git add -A
git commit -q -m @'
VRINGON Vision QA demo

Part-conditioned anomaly detection for finished footwear: silhouette ->
part segmentation -> colorway reference retrieval -> PatchCore -> rule-based
defect typing -> QA verdict.

Ships two acquisition tracks so the measured limits are visible rather than
hidden: free-form public photos (references are different physical units, so
the pass/fail decision does not hold) and a simulated fixed rig (same unit
re-photographed, where it does). Model quality itself is benchmarked on real
factory defects from VisA.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
'@
if ($LASTEXITCODE -ne 0) { Write-Host "nothing new to commit" }
$env:GIT_ASKPASS = ""
git -c credential.helper= push -q --force "https://$Owner`:$token@github.com/$Owner/$Repo.git" main
if ($LASTEXITCODE -ne 0) { throw "push failed ($LASTEXITCODE)" }
Write-Host "pushed"
$ErrorActionPreference = "Stop"
Pop-Location

# 3. Pages from main /docs
$pagesBody = @{ source = @{ branch = "main"; path = "/docs" } }
try {
  Api POST "/repos/$Owner/$Repo/pages" $pagesBody | Out-Null
  Write-Host "pages enabled"
} catch {
  try { Api PUT "/repos/$Owner/$Repo/pages" $pagesBody | Out-Null; Write-Host "pages updated" }
  catch { Write-Host "pages already configured" }
}
$p = Api GET "/repos/$Owner/$Repo/pages" $null
Write-Host "URL: $($p.html_url)"
Write-Host "status: $($p.status)"

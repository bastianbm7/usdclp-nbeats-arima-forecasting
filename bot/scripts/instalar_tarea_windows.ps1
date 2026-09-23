<#
.SYNOPSIS
    Registra (o elimina) la tarea programada de Windows que corre el bot CADA HORA.

.DESCRIPTION
    Por qué cada hora y no una vez al día:
    La vela diaria cierra a las 17:00 de Nueva York. La diferencia horaria Chile <-> Nueva
    York cambia varias veces al año (los dos países cambian de horario en fechas distintas),
    así que una hora fija en Chile a veces caería antes del cierre. Corriendo cada hora al
    minuto :20, el propio bot decide si hay una vela nueva que procesar (y si ya pasaron
    MINUTOS_ESPERA_CIERRE desde el cierre). Las corridas sin nada nuevo terminan en
    segundos con estado 'ya_procesada'. Como Chile y Nueva York difieren en horas enteras,
    el minuto :20 local es también el :20 de Nueva York -> se opera ~17:20 NY todos los
    días hábiles, sin importar el horario de verano.

    La tarea corre con tu usuario, solo con sesión iniciada (no pide contraseña). Si el
    PC está apagado o suspendido, la tarea se ejecuta al volver (StartWhenAvailable) y el
    bot procesa la última vela disponible.

.EXAMPLE
    # Desde PowerShell, en la carpeta bot\ :
    powershell -ExecutionPolicy Bypass -File .\scripts\instalar_tarea_windows.ps1
    powershell -ExecutionPolicy Bypass -File .\scripts\instalar_tarea_windows.ps1 -Desinstalar
#>
param(
    [string]$NombreTarea = "PaperBot_CobreAUD",
    [int]$Minuto = 20,
    [switch]$Desinstalar
)

$ErrorActionPreference = "Stop"

if ($Desinstalar) {
    if (Get-ScheduledTask -TaskName $NombreTarea -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $NombreTarea -Confirm:$false
        Write-Host "Tarea '$NombreTarea' eliminada."
    } else {
        Write-Host "La tarea '$NombreTarea' no existe."
    }
    return
}

$BotDir = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $BotDir ".venv\Scripts\python.exe"
$Script = Join-Path $BotDir "run_diario.py"

if (-not (Test-Path $Python)) { throw "No existe $Python. Crea el venv primero (ver README)." }
if (-not (Test-Path $Script)) { throw "No existe $Script." }
if (-not (Test-Path (Join-Path $BotDir ".env"))) {
    Write-Warning "No existe bot\.env: el bot correrá con los valores por defecto (MODO=simulado)."
}

$Accion = New-ScheduledTaskAction -Execute $Python -Argument "`"$Script`"" -WorkingDirectory $BotDir

# Disparador diario a las 00:MM que se repite cada 1 hora durante 23:59 -> corre cada hora.
$Inicio = (Get-Date).Date.AddMinutes($Minuto)
$Disparador = New-ScheduledTaskTrigger -Daily -At $Inicio
$Repeticion = New-ScheduledTaskTrigger -Once -At $Inicio `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 23 -Minutes 59)
$Disparador.Repetition = $Repeticion.Repetition

$Ajustes = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $NombreTarea -Action $Accion -Trigger $Disparador `
    -Settings $Ajustes -Principal $Principal `
    -Description "Bot de paper trading (DEMO) cobre->AUD. Corre cada hora al minuto :$Minuto; el bot decide si hay vela nueva." `
    -Force | Out-Null

Write-Host "Tarea '$NombreTarea' registrada: cada hora al minuto :$Minuto."
Write-Host "  Python : $Python"
Write-Host "  Script : $Script"
Write-Host "  Logs   : $(Join-Path $BotDir 'logs')"
Write-Host "Probar ahora:  Start-ScheduledTask -TaskName $NombreTarea"
Write-Host "Ver estado  :  Get-ScheduledTaskInfo -TaskName $NombreTarea"

$action = New-ScheduledTaskAction -Execute "D:\projects\mt5_python_ea_suite\run\start_all.bat" -WorkingDirectory "D:\projects\mt5_python_ea_suite\run"
$trigger = New-ScheduledTaskTrigger -AtLogon -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 0)
Register-ScheduledTask -TaskName "MT5_EA_Suite_AutoStart" -Action $action -Trigger $trigger -Settings $settings -Description "MT5 + Proxy Server AutoStart" -Force
Write-Host "Done: MT5_EA_Suite_AutoStart registered"

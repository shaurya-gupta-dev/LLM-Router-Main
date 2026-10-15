' LLM Router - Silent Background Daemon Launcher
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Current script directory
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strPath

' Pythonw executable in virtualenv
strPythonw = strPath & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(strPythonw) Then
    strPythonw = "pythonw.exe"
End If

' Run headless without console window (0 = hidden, False = don't wait)
strCmd = """" & strPythonw & """ """ & strPath & "\run_daemon.py"""
WshShell.Run strCmd, 0, False

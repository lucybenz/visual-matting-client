@echo off
cd /d "%~dp0\.."
dotnet run --project native_matting_client\NativeMattingClient

@echo off
echo ============================================================
echo COMPILING OT_CORE.CPP WITH PARALLEL RESTARTS
echo ============================================================
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "d:\VSCode\curriculum_paper\optimaldiscretemenu"
echo.
echo Compiling with OpenMP for parallel restarts...
echo Deleting old DLL first...
if exist ot_core.dll del ot_core.dll
if exist ot_core.obj del ot_core.obj
echo.
cl /O2 /openmp /EHsc /LD /std:c++17 /I"C:\eigen-3.4.1\eigen-3.4.1" ot_core.cpp /link /OUT:ot_core.dll 2>&1
echo.
echo ============================================================
echo DONE! New ot_core.dll has PARALLEL RESTARTS
echo ============================================================
pause

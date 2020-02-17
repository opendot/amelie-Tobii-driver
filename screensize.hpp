#include "stdafx.h";
#include <atlstr.h>;
#include <iostream>;
#include <SetupApi.h>;
#include <cfgmgr32.h>;   // for MAX_DEVICE_ID_LEN
#pragma comment(lib, "setupapi.lib")

#define NAME_SIZE 128

const GUID GUID_CLASS_MONITOR = { 0x4d36e96e, 0xe325, 0x11ce, 0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18 };

CString Get2ndSlashBlock(const CString& sIn)
{
	int FirstSlash = sIn.Find(_T('\\'));
	CString sOut = sIn.Right(sIn.GetLength() - FirstSlash - 1);
	FirstSlash = sOut.Find(_T('\\'));
	sOut = sOut.Left(FirstSlash);
	return sOut;
}

// Assumes hEDIDRegKey is valid
bool GetMonitorSizeFromEDID(const HKEY hEDIDRegKey, short& WidthMm, short& HeightMm)
{
	BYTE EDIDdata[1024];
	DWORD edidsize = sizeof(EDIDdata);

	if (ERROR_SUCCESS != RegQueryValueEx(hEDIDRegKey, _T("EDID"), NULL, NULL, EDIDdata, &edidsize))
		return false;
	WidthMm = ((EDIDdata[68] & 0xF0) << 4) + EDIDdata[66];
	HeightMm = ((EDIDdata[68] & 0x0F) << 8) + EDIDdata[67];

	return true; // valid EDID found
}

bool GetSizeForDevID(const CString& TargetDevID, short& WidthMm, short& HeightMm)
{
	HDEVINFO devInfo = SetupDiGetClassDevsEx(
		&GUID_CLASS_MONITOR, //class GUID
		NULL, //enumerator
		NULL, //HWND
		DIGCF_PRESENT | DIGCF_PROFILE, // Flags //DIGCF_ALLCLASSES|
		NULL, // device info, create a new one.
		NULL, // machine name, local machine
		NULL);// reserved

	if (NULL == devInfo)
		return false;

	bool bRes = false;

	for (ULONG i = 0; ERROR_NO_MORE_ITEMS != GetLastError(); ++i)
	{
		SP_DEVINFO_DATA devInfoData;
		memset(&devInfoData, 0, sizeof(devInfoData));
		devInfoData.cbSize = sizeof(devInfoData);

		if (SetupDiEnumDeviceInfo(devInfo, i, &devInfoData))
		{
			TCHAR Instance[MAX_DEVICE_ID_LEN];
			SetupDiGetDeviceInstanceId(devInfo, &devInfoData, Instance, MAX_PATH, NULL);

			CString sInstance(Instance);
			if (-1 == sInstance.Find(TargetDevID))
				continue;

			HKEY hEDIDRegKey = SetupDiOpenDevRegKey(devInfo, &devInfoData,
				DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ);

			if (!hEDIDRegKey || (hEDIDRegKey == INVALID_HANDLE_VALUE))
				continue;

			bRes = GetMonitorSizeFromEDID(hEDIDRegKey, WidthMm, HeightMm);

			RegCloseKey(hEDIDRegKey);
		}
	}
	SetupDiDestroyDeviceInfoList(devInfo);
	return bRes;
}

HMONITOR  g_hMonitor;

BOOL CALLBACK MyMonitorEnumProc(
	_In_  HMONITOR hMonitor,
	_In_  HDC hdcMonitor,
	_In_  LPRECT lprcMonitor,
	_In_  LPARAM dwData
)

{
	// Use this function to identify the monitor of interest: MONITORINFO contains the Monitor RECT.
	MONITORINFOEX mi;
	mi.cbSize = sizeof(MONITORINFOEX);

	GetMonitorInfo(hMonitor, &mi);
	OutputDebugString(mi.szDevice);

	// For simplicity, we set the last monitor to be the one of interest
	g_hMonitor = hMonitor;

	return TRUE;
}

BOOL DisplayDeviceFromHMonitor(HMONITOR hMonitor, DISPLAY_DEVICE& ddMonOut)
{
	MONITORINFOEX mi;
	mi.cbSize = sizeof(MONITORINFOEX);
	GetMonitorInfo(hMonitor, &mi);

	DISPLAY_DEVICE dd;
	dd.cb = sizeof(dd);
	DWORD devIdx = 0; // device index

	CString DeviceID;
	bool bFoundDevice = false;
	while (EnumDisplayDevices(0, devIdx, &dd, 0))
	{
		devIdx++;
		if (0 != _tcscmp(dd.DeviceName, mi.szDevice))
			continue;

		DISPLAY_DEVICE ddMon;
		ZeroMemory(&ddMon, sizeof(ddMon));
		ddMon.cb = sizeof(ddMon);
		DWORD MonIdx = 0;

		while (EnumDisplayDevices(dd.DeviceName, MonIdx, &ddMon, 0))
		{
			MonIdx++;

			ddMonOut = ddMon;
			return TRUE;

			ZeroMemory(&ddMon, sizeof(ddMon));
			ddMon.cb = sizeof(ddMon);
		}

		ZeroMemory(&dd, sizeof(dd));
		dd.cb = sizeof(dd);
	}

	return FALSE;
}

std::pair<short, short> getScreenPhysicalSize()
{
	// Identify the HMONITOR of interest via the callback MyMonitorEnumProc
	EnumDisplayMonitors(NULL, NULL, MyMonitorEnumProc, NULL);

	DISPLAY_DEVICE ddMon;
	if (FALSE == DisplayDeviceFromHMonitor(g_hMonitor, ddMon)) {
		std::pair <short, short> dims(0, 0);
		return dims;
	}

	CString DeviceID;
	DeviceID.Format(_T("%s"), ddMon.DeviceID);
	DeviceID = Get2ndSlashBlock(DeviceID);

	short WidthMm, HeightMm;
	bool bFoundDevice = GetSizeForDevID(DeviceID, WidthMm, HeightMm);

	//std::cout << WidthMm << " -- " << HeightMm;
	std::pair <short, short> dims(WidthMm, HeightMm);
	return dims;
}
//This is a test cpp file
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")
main(){
std::pair<CString,CString> getComputerManufacturerAndModel() {
    // Obtain the initial locator to Windows Management on a particular host computer.
    IWbemLocator *locator = nullptr;
    IWbemServices *services = nullptr;
    auto hResult = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID *)&locator);

    auto hasFailed = [&hResult]() {
        if (FAILED(hResult)) {
            auto error = _com_error(hResult);
            TRACE(error.ErrorMessage());
            TRACE(error.Description().Detach());
            return true;
        }
        return false;
    };

    auto getValue = [&hResult, &hasFailed](IWbemClassObject *classObject, LPCWSTR property) {
        CString propertyValueText = "Not set";
        VARIANT propertyValue;
        hResult = classObject->Get(property, 0, &propertyValue, 0, 0);
        if (!hasFailed()) {
            if ((propertyValue.vt == VT_NULL) || (propertyValue.vt == VT_EMPTY)) {
            } else if (propertyValue.vt & VT_ARRAY) {
                propertyValueText = "Unknown"; //Array types not supported
            } else {
                propertyValueText = propertyValue.bstrVal;
            }
        }
        VariantClear(&propertyValue);
        return propertyValueText;
    };

    CString manufacturer = "Not set";
    CString model = "Not set";
    if (!hasFailed()) {
        // Connect to the root\cimv2 namespace with the current user and obtain pointer pSvc to make IWbemServices calls.
        hResult = locator->ConnectServer(L"ROOT\\CIMV2", nullptr, nullptr, 0, NULL, 0, 0, &services);

        if (!hasFailed()) {
            // Set the IWbemServices proxy so that impersonation of the user (client) occurs.
            hResult = CoSetProxyBlanket(services, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr, RPC_C_AUTHN_LEVEL_CALL,
                RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE);

            if (!hasFailed()) {
                IEnumWbemClassObject* classObjectEnumerator = nullptr;
                hResult = services->ExecQuery(L"WQL", L"SELECT * FROM Win32_ComputerSystem", WBEM_FLAG_FORWARD_ONLY |
                    WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &classObjectEnumerator);
                if (!hasFailed()) {
                    IWbemClassObject *classObject;
                    ULONG uReturn = 0;
                    hResult = classObjectEnumerator->Next(WBEM_INFINITE, 1, &classObject, &uReturn);
                    if (uReturn != 0) {
                        manufacturer = getValue(classObject, (LPCWSTR)L"Manufacturer");
                        model = getValue(classObject, (LPCWSTR)L"Model");
                    }
                    classObject->Release();
                }
                classObjectEnumerator->Release();
            }
        }
    }

    if (locator) {
        locator->Release();
    }
    if (services) {
        services->Release();
    }
    CoUninitialize();
    return { manufacturer, model };
}

printf(manufacturer, model)

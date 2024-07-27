import ctypes


def read_memory(pid, address, size):
    PROCESS_ALL_ACCESS = 0x1F0FFF
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not process:
        raise Exception("Could not open process: %s" % pid)

    buffer = (ctypes.c_char * size)()
    bytesRead = ctypes.c_size_t()
    if not kernel32.ReadProcessMemory(process, ctypes.c_void_p(address), buffer, ctypes.sizeof(buffer),
                                      ctypes.byref(bytesRead)):
        raise Exception("Could not read memory at address: %s" % hex(address))

    kernel32.CloseHandle(process)
    return buffer.raw


def read_index(pid, address):
    PROCESS_ALL_ACCESS = 0x1F0FFF
    kernel32 = ctypes.windll.kernel32
    process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not process:
        raise Exception("Could not open process: %s" % pid)

    buffer = ctypes.c_size_t()
    bytesRead = ctypes.c_size_t()
    if not kernel32.ReadProcessMemory(process, ctypes.c_void_p(address), ctypes.byref(buffer), ctypes.sizeof(buffer),
                                      ctypes.byref(bytesRead)):
        raise Exception("Could not read memory at address: %s" % hex(address))

    kernel32.CloseHandle(process)
    return buffer.value


def main():
    # inputs:
    pid = 292964
    data_address = 0x7ffe252a2273
    index_address = 0x2e35c213740

    entry_size = 128
    num_entries = 32
    total_size = entry_size * num_entries

    raw_data = read_memory(pid, data_address, total_size)
    current_index = read_index(pid, index_address)

    entries = [raw_data[i * entry_size:(i + 1) * entry_size] for i in range(num_entries)]

    ordered_entries = entries[current_index:] + entries[:current_index]

    for i, entry in enumerate(ordered_entries):
        entry_str = entry.decode('utf-8', 'ignore').rstrip('\x00')
        print(f"Entry {i}: {entry_str}")


if __name__ == "__main__":
    main()

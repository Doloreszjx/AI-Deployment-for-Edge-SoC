import subprocess
import re


def check_vnni_support():
    try:
        # Linux/Mac
        output = subprocess.check_output(['lscpu'], text=True)
        if 'avx512_vnni' in output.lower():
            return "AVX512-VNNI"
        elif 'avx_vnni' in output.lower():
            return "AVX-VNNI"
    except:
        pass

    try:
        # Windows
        print('windows checking')
        output = subprocess.check_output(['wmic', 'cpu', 'get', 'caption'], text=True)
        cpu_info = output.lower()
        if 'cascade lake' in cpu_info or 'ice lake' in cpu_info:
            return "Likely supports AVX512-VNNI"
        elif 'alder lake' in cpu_info or 'raptor lake' in cpu_info:
            return "Likely supports AVX-VNNI"
    except:
        pass

    return "Unknown or not supported"


print(f"VNNI支持: {check_vnni_support()}")
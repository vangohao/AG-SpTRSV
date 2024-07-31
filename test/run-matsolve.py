import json
import re
import subprocess
import sys

int_max = 2147483647
args = sys.argv
parsed_data = {}
run_name = args[1]
executable_path = "./" + run_name


def run_test(arguments):
    global parsed_data
    command = [executable_path] + arguments
    result = subprocess.run(command, capture_output=True, text=True)
    stdout = result.stdout.split("\n")
    if result.stderr:
        return True

    stencils = {0: "stencilstar", 1: "stencilbox", 2: "stencilstarfill1"}
    stencil, stencil_width, dof, m, n, p = map(int, arguments)
    problem = stencils[stencil] + ",width=%d" % stencil_width
    dof_str = "dof=%d" % dof
    lower_pattern = re.compile(r"\s+Lower:(\d+),([\d\.]+),([\d\.]+),([\d\.]+)")
    err_pattern = re.compile(r".*out of memory.*")
    if problem not in parsed_data:
        parsed_data[problem] = {}
    if dof_str not in parsed_data[problem]:
        parsed_data[problem][dof_str] = []
    for line in stdout:
        err_match = err_pattern.match(line)
        if err_match:
            return True
        lower_match = lower_pattern.match(line)
        if lower_match:
            lower_data = tuple(map(float, lower_match.groups()))
            parsed_data[problem][dof_str].append(
                {"mesh_size": (m, n, p), "lower": lower_data}
            )
    return False


def generate_test(stencil_type, width, dof):
    k = 32
    while k**3 < int_max:
        if run_test(list(map(str, [stencil_type, width, dof, k, k, k]))):
            break
        print("mesh size:", k)
        k += 16


generate_test(0, 0, 1)
generate_test(0, 1, 1)
generate_test(1, 0, 1)
generate_test(2, 0, 1)
generate_test(0, 0, 4)
generate_test(0, 1, 4)
generate_test(1, 0, 4)
generate_test(2, 0, 4)
with open("./results/" + run_name + ".json", "w", encoding="utf-8") as fout:
    json.dump(parsed_data, fout, indent=4)

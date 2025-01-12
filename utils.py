import numpy as np
from pysr import PySRRegressor
import sympy
import re
import base64


BINOPS = ["+", "-", "max", "min"]

def remove_constant_and_equivalent(rams):
    ncells = len(rams[0])
    constants = []
    equivalents = {}
    for i in range(ncells):
        if len(np.unique(all_states[:, i])) == 1:
            constants.append(i)
        else:
            stop = False
            for j in range(i):
                if stop:
                    break
                if np.all(rams[:, j] == rams[:, i]):
                    equivalents[j] = equivalents.get(j, []) + [i]
                    stop = True

    to_remove = np.unique(constants + sum(list(equivalents.values()), []))
    rams_mapping = [i for i in list(range(ncells)) if i not in to_remove]
    return np.array(rams_mapping), equivalents

def get_model(l1_loss=True, min_val=None, max_val=None, binops=BINOPS, vnames=[]):
    if l1_loss:
        loss = "L1DistLoss()"
    else:
        loss = "L2DistLoss()"

    un_ops = []
    extra_sympy_mappings = {}
    if min_val is not None:
        f = "max_" + str(min_val)
        f = f.replace("-", "minus_")
        un_ops.append(f + "(x) = max(x, " + str(min_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Max(min_val, x)
    if max_val is not None:
        f = "min_" + str(max_val)
        f = f.replace("-", "minus_")
        un_ops.append(f + "(x) = min(x, " + str(max_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Min(max_val, x)

    return PySRRegressor(
        niterations = 50,  # < Increase me for better results
        maxsize = 10,
        binary_operators = binops,
        unary_operators = un_ops,
        extra_sympy_mappings = extra_sympy_mappings,
        elementwise_loss = loss,
        complexity_of_variables=2,
        temp_equation_file=True,  # Don't write final or intermediate CSVs
    )

def extend_with_signed_rams(rams):
    nstates, ncells = rams.shape
    extended_rams = np.zeros((nstates, 2 * ncells), dtype=int)
    extended_rams[:, :ncells] = rams
    for i, ram in enumerate(rams):
        signed_ram = ram.astype(np.int8)
        extended_rams[i, ncells:] = signed_ram
    return extended_rams

def replace_vnames(eq):
    eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
    eq = re.sub(r'min\[(\d{1,3})\]\(', r'min(\1, ', eq)
    eq = re.sub(r'max\[(\d{1,3})\]\(', r'max(\1, ', eq)
    try:
        eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
        eq = re.sub(r'min\[(\d{1,3})\]\(', r'min(\1, ', eq)
        eq = re.sub(r'max\[(\d{1,3})\]\(', r'max(\1, ', eq)
        return eq
    except TypeError:
        return eq

def replace_float_with_int_if_close(s):
    pattern = re.compile(r'-?\d+(\.\d+)?')  # matches possible signed floats/ints

    def maybe_int(match):
        original_text = match.group()
        value = float(original_text)
        # Check if it's "close" to an integer:
        if abs(value - round(value)) < 1e-3:
            # Replace with integer
            return str(int(round(value)))
        else:
            # Keep the original float
            return original_text

    return pattern.sub(maybe_int, s)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/" + image_path.split(".")[-1]  # Infer MIME type from file extension
        return f"data:{mime_type};base64,{encoded_string}"

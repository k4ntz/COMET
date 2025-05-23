import numpy as np
from pysr import PySRRegressor
import sympy
import re
import base64
import sympy as sp
import json
import pydot

BINOPS = ["+", "-", "max", "min"]

RAM_PATTERN = r'ram\[(\d{1,3})\]'
ACT_PATTERN = r'act\[(\d{1,3})\]'
SRAM_RAM_PATTERN = r'\b(s?ram)\[(\d{1,3})\]'

def remove_constant_and_equivalent(rams):
    ncells = len(rams[0])
    constants = []
    equivalents = {}
    for i in range(ncells):
        if len(np.unique(rams[:, i])) == 1:
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

def get_model(l1_loss=True, min_val=None, max_val=None, mod_max=None,
              binops=BINOPS, complexity_of_vars=2):
    if l1_loss:
        loss = "L1DistLoss()"
    else:
        loss = "L2DistLoss()"

    un_ops = []
    extra_sympy_mappings = {}
    if min_val is not None:
        f = "above_" + str(min_val)
        f = f.replace("-", "minus_")
        un_ops.append(f + "(x) = max(x, " + str(min_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Max(min_val, x)
    if max_val is not None:
        f = "below_" + str(max_val)
        f = f.replace("-", "minus_")
        un_ops.append(f + "(x) = min(x, " + str(max_val) + ")")
        extra_sympy_mappings[f] = lambda x: sympy.Min(max_val, x)
    if mod_max is not None:
        f = "mod_" + str(mod_max)
        un_ops.append(f + "(x) = mod(" + str(mod_max) + ", x)")
        extra_sympy_mappings[f] = lambda x: sympy.Mod(x, mod_max)


    return PySRRegressor(
        population_size=50,
        niterations = 50,  # < Increase me for better results
        maxsize = 10,
        binary_operators = binops,
        unary_operators = un_ops,
        extra_sympy_mappings = extra_sympy_mappings,
        elementwise_loss = loss,
        complexity_of_variables = complexity_of_vars,
        temp_equation_file=True,  # Don't write final or intermediate CSVs
    )

# def extend_with_signed_rams(rams):
#     nstates, ncells = rams.shape
#     extended_rams = np.zeros((nstates, 2 * ncells), dtype=int)
#     extended_rams[:, :ncells] = rams
#     for i, ram in enumerate(rams):
#         signed_ram = ram.astype(np.int8)
#         extended_rams[i, ncells:] = signed_ram
#     return extended_rams

def replace_vnames(eq):
    try:
        eq = re.sub(r'_(\d{1,3})', r'[\1]', eq)
        eq = re.sub(r'below\[(\d{1,3})\]\(', r'min(\1, ', eq)
        eq = re.sub(r'above\[(\d{1,3})\]\(', r'max(\1, ', eq)
        eq = re.sub(r'below_minus\[(\d{1,3})\]\(', r'min(-\1, ', eq)
        eq = re.sub(r'above_minus\[(\d{1,3})\]\(', r'max(-\1, ', eq)
        eq = re.sub(r'mod\[(\d+)\]\((.+)\)', r'mod(\2, \1)', eq)
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

def simplify_equation_with_arrays(rhs):
    rhs = re.sub(r"(\w+)\[(\d+)\]", r"\1_\2", rhs)
    
    # Split into left-hand and right-hand sides
    rhs = sp.sympify(rhs.strip())  # Parse RHS

    # Simplify RHS and round every number to an integer
    def round_constants(expr):
        return expr.xreplace({n: int(round(n)) for n in expr.atoms(sp.Number)})

    simplified_rhs = round_constants(sp.simplify(rhs))
    

    # Convert identifiers back to array-style notation
    equation_str_simplified = sp.printing.pretty(simplified_rhs, use_unicode=False)
    equation_str_simplified = re.sub(r"(\w+)_(\d+)", r"\1[\2]", equation_str_simplified)

    return equation_str_simplified


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/" + image_path.split(".")[-1]  # Infer MIME type from file extension
        return f"data:{mime_type};base64,{encoded_string}"

    
def eq_name(ram_idx, next):
    eqname = f"ram[{ram_idx}]"
    if next:
        eqname = 'n' + eqname
    return eqname


def find_connected_rams(eq):
    if "ram" in eq:
        return re.findall(RAM_PATTERN, eq)


def convert_to_svg(network, filename):
    """Convert a Pyvis network to a DOT graph and save it as an SVG file."""
    graph_json = network.to_json()

    # Load JSON data
    data = json.loads(graph_json)

    # Extract nodes and edges
    nodes = eval(data["nodes"])  # Convert stringified list to Python list
    edges = eval(data["edges"])  # Convert stringified list to Python list

    # Initialize DOT graph
    dot_graph = "digraph G {\n"

    # Add nodes
    for node in nodes:
        node_id = node["id"]
        label = node.get("label", node_id)
        color = node.get("color", "#000000")
        dot_graph += f'    "{node_id}" [label="{label}", color="{color}"];\n'

    # Add edges
    for edge in edges:
        from_node = edge["from"]
        to_node = edge["to"]
        title = edge.get("title", "")
        dot_graph += f'    "{from_node}" -> "{to_node}" [label="{title}"];\n'

    # Close DOT graph
    dot_graph += "}"

    graphs = pydot.graph_from_dot_data(dot_graph)
    svg_string = graphs[0].create_svg()
    with open(f"{filename}", "wb") as file:
        file.write(svg_string)
    print(f"SVG file saved as {filename}")
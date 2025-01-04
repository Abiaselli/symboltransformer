from sympy import symbols, Matrix
import json
import tkinter as tk
from tkinter import filedialog
import torch
from sympy import Function, exp, log
import re

# Step 1: Define a symbolic tensor class
class SymbolicTensor:
    def __init__(self, shape, name="x", expression=None):
        self.shape = shape
        self.values = expression if expression else Matrix([[symbols(f"{name}{i}{j}") for j in range(shape[1])] for i in range(shape[0])])

    def __repr__(self):
        return str(self.values)

# Step 2: Define basic operations for the symbolic tensors
class SymbolicOperations:
    @staticmethod
    def matmul(tensor1, tensor2):
        assert tensor1.shape[1] == tensor2.shape[0], "Shapes do not match for matmul"
        new_expression = tensor1.values * tensor2.values
        return SymbolicTensor((tensor1.shape[0], tensor2.shape[1]), expression=new_expression), new_expression

    @staticmethod
    def add(tensor1, tensor2):
        assert tensor1.shape == tensor2.shape, "Shapes do not match for addition"
        new_expression = tensor1.values + tensor2.values
        return SymbolicTensor(tensor1.shape, expression=new_expression), new_expression

    @staticmethod
    def einsum(equation, *tensors):
        # Parse the equation
        input_indices, output_indices = equation.split("->")
        input_indices = input_indices.split(",")

        # Ensure the number of tensors matches
        assert len(input_indices) == len(tensors), "Number of tensors and indices must match"

        # Build the symbolic result
        result = 0
        for idx_combination in zip(*[range(tensor.shape[0]) for tensor in tensors]):
            term = 1
            for i, tensor in enumerate(tensors):
                term *= tensor.values[idx_combination[i]]
            result += term

        # Simplified result shape (for example purposes)
        result_shape = (1, 1) if output_indices else tensors[0].shape

        return SymbolicTensor(result_shape, expression=result), result

    @staticmethod
    def sum(tensor, axis=None):
        if axis is None:
            summed = tensor.values.applyfunc(lambda x: symbols(f"sum({x})"))
        else:
            summed = tensor.values.sum(axis=axis)
        return SymbolicTensor((1, 1), expression=summed), summed

    @staticmethod
    def sigmoid(tensor):
        sig_expression = tensor.values.applyfunc(lambda x: 1 / (1 + exp(-x)))
        return SymbolicTensor(tensor.shape, expression=sig_expression), sig_expression

    @staticmethod
    def silu(tensor):
        silu_expression = tensor.values.applyfunc(lambda x: x / (1 + exp(-x)))
        return SymbolicTensor(tensor.shape, expression=silu_expression), silu_expression

    @staticmethod
    def softmax(tensor, axis=0):
        exp_tensor = tensor.values.applyfunc(exp)
        sum_exp = exp_tensor.sum(axis=axis)
        softmax_expression = exp_tensor / sum_exp
        return SymbolicTensor(tensor.shape, expression=softmax_expression), softmax_expression

# Step 3: Integration into the Transformer
class SimpleSymbolicTransformer:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Initialize symbolic weights
        self.embedding = SymbolicTensor((embed_dim, vocab_size), name="embed")
        self.attention_weights = SymbolicTensor((embed_dim, embed_dim), name="attn")
        self.feedforward_weights = SymbolicTensor((hidden_dim, embed_dim), name="ff")
        self.output_weights = SymbolicTensor((vocab_size, hidden_dim), name="out")

    def forward(self, input_tensor, sequence_length=1):
        outputs = []

        # Generate sequence logits
        for _ in range(sequence_length):
            # Embedding layer
            embedded, embedded_values = SymbolicOperations.matmul(self.embedding, input_tensor)

            # Attention layer
            attended, attended_values = SymbolicOperations.matmul(self.attention_weights, embedded)
            attended = SymbolicTensor(attended.shape, expression=attended_values)

            # Feedforward layer
            ff_output, ff_values = SymbolicOperations.matmul(self.feedforward_weights, attended)
            ff_output = SymbolicTensor(ff_output.shape, expression=ff_values)

            # Output layer
            output, output_values = SymbolicOperations.matmul(self.output_weights, ff_output)
            output = SymbolicTensor(output.shape, expression=output_values)

            outputs.append(str(output_values))

        return {
            "sequence_logits": outputs
        }

# Batch Optimization Helper
class BatchOptimizer:
    @staticmethod
    def optimize(json_file, target_vocab):
        with open(json_file, "r") as file:
            data = json.load(file)

        optimized_results = {}
        for token, expression in data.items():
            if token.startswith("output"):
                # Example: Replace symbolic variables with target vocab
                optimized_expression = expression
                for vocab_index, vocab_token in enumerate(target_vocab):
                    optimized_expression = optimized_expression.replace(f"input{vocab_index}", vocab_token)

                optimized_results[token] = optimized_expression

        # Save optimized results
        optimized_file = json_file.replace(".json", "_optimized.json")
        with open(optimized_file, "w") as file:
            json.dump(optimized_results, file, indent=4)

        print(f"Optimized results saved to {optimized_file}")

# GUI Application
class SymbolicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symbolic Transformer Viewer")

        # Add buttons
        self.run_button = tk.Button(root, text="Run Transformer", command=self.run_transformer)
        self.run_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Output to File", command=self.save_to_file)
        self.save_button.pack(pady=10)

        self.optimize_button = tk.Button(root, text="Optimize Output File", command=self.optimize_output)
        self.optimize_button.pack(pady=10)

        self.outputs = None

    def run_transformer(self):
        # Use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {device}")

        # Run the symbolic transformer
        vocab_size = 10
        embed_dim = 8
        num_heads = 1
        num_layers = 1
        hidden_dim = 16
        sequence_length = 3  # Example sequence length

        symbolic_transformer = SimpleSymbolicTransformer(vocab_size, embed_dim, num_heads, num_layers, hidden_dim)
        input_tensor = SymbolicTensor((vocab_size, 1), name="input")
        self.outputs = symbolic_transformer.forward(input_tensor, sequence_length=sequence_length)

    def save_to_file(self):
        if self.outputs is None:
            print("Run the transformer first before saving!")
            return

        # Open save file dialog
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.outputs, file, indent=4)
            print(f"Outputs saved to {file_path}")

    def optimize_output(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not file_path:
            print("No file selected for optimization.")
            return

        target_vocab = [f"token{i}" for i in range(10)]  # Example target vocab
        BatchOptimizer.optimize(file_path, target_vocab)

from sympy import symbols, Symbol, MatrixSymbol, Add, Mul, Function
from typing import List, Union

class SymbolicNode:
    def __init__(self, 
                 name: str, 
                 op_type: str, 
                 inputs: List['SymbolicNode'] = None, 
                 params: dict = None,
                 lazy_expr=None):
        """
        :param name: A unique name or ID for this node.
        :param op_type: The operation type, e.g. "matmul", "add", "sigmoid".
        :param inputs: List of other SymbolicNodes that feed into this node.
        :param params: A dict of parameter references (weights, biases, etc.).
        :param lazy_expr: A Sympy object or placeholder to represent the expression.
        """
        self.name = name
        self.op_type = op_type
        self.inputs = inputs if inputs else []
        self.params = params if params else {}
        self.lazy_expr = lazy_expr  # e.g. a MatrixSymbol or partial expression
        self._expanded_cache = None  # cache for fully expanded expression

    def __repr__(self):
        return f"<SymbolicNode {self.name} op={self.op_type}>"

    def expand_expression(self):
        """
        Recursively expand or build the full Sympy expression 
        from the node's inputs & params. Caches the result.
        """
        if self._expanded_cache is not None:
            return self._expanded_cache
        
        # Base case: If no inputs, e.g. "leaf" parameter node
        if not self.inputs:
            # If we have lazy_expr as a MatrixSymbol or literal, just return it:
            self._expanded_cache = self.lazy_expr
            return self._expanded_cache

        # Otherwise, recursively expand inputs
        expanded_inputs = [inp.expand_expression() for inp in self.inputs]

        if self.op_type == "matmul":
            # For simplicity, assume the first input is e.g. a param (MatrixSymbol)
            # and the second is the next node's expression
            # Then do something like expanded_inputs[0] * expanded_inputs[1]
            if len(expanded_inputs) != 2:
                raise ValueError("matmul requires exactly 2 inputs.")
            self._expanded_cache = expanded_inputs[0] * expanded_inputs[1]

        elif self.op_type == "add":
            # Summation of all inputs
            expr = expanded_inputs[0]
            for e in expanded_inputs[1:]:
                expr = expr + e
            self._expanded_cache = expr

        elif self.op_type == "sigmoid":
            # 1 / (1 + e^-x)
            if len(expanded_inputs) != 1:
                raise ValueError("sigmoid requires exactly 1 input.")
            x = expanded_inputs[0]
            self._expanded_cache = 1 / (1 + Symbol('e')**(-x))

        # More ops: "softmax", "silu", "rmsnorm", etc.
        # or do partial expansions only; keep them in function form, e.g. Function('Softmax')(expanded_inputs)

        return self._expanded_cache

    def invalidate_cache(self):
        """
        If we edit parameters or structure, we should invalidate the cached expansions
        so the next expand_expression() will recalc.
        """
        self._expanded_cache = None
        for inp in self.inputs:
            inp.invalidate_cache()

    def substitute_params(self, **param_values):
        """
        Example method to do sympy.subs() on known parameters if they are symbolic.
        param_values: e.g. embed00=3.14, embed01=2.72, etc.
        """
        expr = self.expand_expression()
        if expr is not None:
            return expr.subs(param_values)
        return expr

class SymbolicGraph:
    def __init__(self):
        self.nodes = {}
        self.root_node = None  # The final output node, e.g. "logits"

    def add_node(self, node: SymbolicNode):
        self.nodes[node.name] = node

    def set_root(self, node: SymbolicNode):
        self.root_node = node

    def expand_all(self):
        if self.root_node:
            return self.root_node.expand_expression()
        return None

    def invalidate_all(self):
        if self.root_node:
            self.root_node.invalidate_cache()

    def substitute_all(self, **param_values):
        """
        Substitutes values across the entire graph, returns a new expression for root.
        """
        if self.root_node:
            return self.root_node.substitute_params(**param_values)
        return None


import sympy

def build_symbolic_transformer_graph(vocab_size=10, embed_dim=8):
    graph = SymbolicGraph()

    # Create a leaf param node: embedding matrix
    # Let's represent it as a MatrixSymbol with shape (embed_dim, vocab_size)
    embed_weight = SymbolicNode(
        name="embed_weight",
        op_type="leaf",
        lazy_expr=sympy.MatrixSymbol("EmbedW", embed_dim, vocab_size)
    )
    graph.add_node(embed_weight)

    # Create an input node (symbolic)
    input_node = SymbolicNode(
        name="input_symbol",
        op_type="leaf",
        lazy_expr=sympy.MatrixSymbol("Input", vocab_size, 1)
    )
    graph.add_node(input_node)

    # Embedding op: matmul(EmbedW, Input)
    embed_op = SymbolicNode(
        name="embedded",
        op_type="matmul",
        inputs=[embed_weight, input_node]
    )
    graph.add_node(embed_op)

    # Suppose we want a Sigmoid afterwards
    sigmoid_node = SymbolicNode(
        name="embed_sigmoid",
        op_type="sigmoid",
        inputs=[embed_op]
    )
    graph.add_node(sigmoid_node)

    # Mark the final node as root
    graph.set_root(sigmoid_node)
    return graph


import tkinter as tk
from tkinter import ttk

class SymbolicGraphGUI(tk.Frame):
    def __init__(self, master, graph: SymbolicGraph):
        super().__init__(master)
        self.graph = graph
        self.pack(fill="both", expand=True)

        # List nodes
        self.nodes_listbox = tk.Listbox(self, height=10)
        self.nodes_listbox.pack(side="left", fill="y")
        for node_name in self.graph.nodes:
            self.nodes_listbox.insert("end", node_name)

        # Button to "Expand Expression" for the selected node
        self.expand_button = tk.Button(self, text="Expand Node", command=self.expand_selected_node)
        self.expand_button.pack(side="top")

        # Text box to display expansions
        self.expr_text = tk.Text(self, wrap="word", width=80, height=20)
        self.expr_text.pack(side="right", fill="both", expand=True)

    def expand_selected_node(self):
        selection = self.nodes_listbox.curselection()
        if selection:
            idx = selection[0]
            node_name = self.nodes_listbox.get(idx)
            node = self.graph.nodes[node_name]

            # Expand
            expanded_expr = node.expand_expression()
            self.expr_text.delete("1.0", tk.END)
            self.expr_text.insert(tk.END, f"Node: {node_name}\n{str(expanded_expr)}\n")

def run_symbolic_gui(graph):
    root = tk.Tk()
    root.title("Symbolic Graph GUI")
    app = SymbolicGraphGUI(root, graph)
    root.mainloop()

if __name__ == "__main__":
    g = build_symbolic_transformer_graph()
    run_symbolic_gui(g)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolicApp(root)
    root.mainloop()

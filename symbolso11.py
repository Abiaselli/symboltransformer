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

#if __name__ == "__main__":
 #   g = build_symbolic_transformer_graph()
  #  run_symbolic_gui(g)


if __name__ == "__main__":
    g = build_symbolic_transformer_graph()
    expanded_expr = g.expand_all()  # triggers lazy expansions
    print("Final expanded expression:\n", expanded_expr)

    # Example partial param substitution
    # e.g. "EmbedW[0,0]" could become a numeric constant if we made them individual symbols
    # (In this minimal example, it's just a MatrixSymbol.)
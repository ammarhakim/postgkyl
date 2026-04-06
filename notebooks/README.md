postgkyl notebooks 
==================

Notebooks using Marimo or Jupyter are useful collections of operations to pre- or post-process Gkeyll simulations, or better understand their algorithms. This folder contains some such notebooks. 
File convention: We use the `*.mo.py` extension for Marimo notebooks, which are Python scripts that can be executed in a notebook-like environment.

- twist_shift.mo.py: Analyze the impact of twist-shift boundary conditions on mode shearing and aliasing, and test possible filtering strategies to mitigate these effects.


### How to use Marimo notebooks
First install Marimo, e.g. `pip install marimo`. Then you can run the notebooks with `marimo run <notebook_name>`. For example, to run the twist_shift notebook, you would use `marimo run twist_shift.mo.py`. This will execute the notebook in a web browser, allowing you to interact with the code and visualize the results.
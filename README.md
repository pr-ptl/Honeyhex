# Honeyhex



Pattern formation in closely packed structures.



This project simulates how collections of deformable circular cells evolve over time using the Immersed Boundary Method. Each cell is represented by a closed curve discretized into Lagrangian points. These curves push on an Eulerian grid through a smooth delta kernel, and the grid returns forces back to the boundaries, details to which are found in the paper of Jeong2018.



The goal is to explore how simple mechanical rules can produce complex structures like honeycomb patterns, grain-like foams, and fluid-like surfaces.


Convex hull experiment to see if we can use a differentiable surrogate to force the model to learn a projection that preserves the convex hull structure while also projecting an epsilon vicinity of the data to the convex hull. 

- differentiable_convex_hull.py: The returned scores are the distances from the query point to the predicted point, calculated using weighted average of the vertices.
  - **unconstrained_optimization**: Uses an optimization loop to find the best weights that minimize the distance from the query to the convex hull defined by the vertices.
  - **kernel_reconstruction**: Uses a kernel-based approach to compute the weights based on distances from the query to the vertices.

- convex_hull_experiment.py: To empirically evaluate the performance of the two methods (unconstrained optimization and kernel-based prediction)
  - scalar dataset projected to higher dimension with a random, linear mapping.
  - any linear projected will preserve the convex hull structure.
  
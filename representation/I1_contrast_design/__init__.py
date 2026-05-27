"""
I.1 Experimental Contrast Design

Keep prompt template and image fixed; change exactly one concept at a time.

Persona track   : vary gender | emotion | cognitive load | job | country | age
Interestingness : compare low-interest vs high-interest activations with fixed persona

Exports:
  collect.run_experiment     — main collection loop (requires GPU + model)
  collect.model_response     — single image inference + activation capture
  load.load_activation_results — load stored .npy result files
  load.filter_by_label_support — drop labels below minimum support threshold
"""

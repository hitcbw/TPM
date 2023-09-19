# Note 
We simplify the training logic, which don't follow config-based and function-based code. We place all codes out of the function except public code blocks.
# File Tree
- loss
    - the folder for loss. We only place our used label smoothing loss here.
- models
    - the folder for models. Place your new model file here, then rename the main class as "Model". The runner would automatically load the model with configuration (parameter: args)
- TPM_MES_lara.py
    - the main runner.
        - private block: YOU ONLY NEED TO CHANGE DATASET NAME & MODEL NAME HERE.
        - public block: load yaml config & find available cuda. We only utilize one GPU.
        - public block: build dataset and build loss
        - public block: train/eval controller    
- README.md
# Acknowledgement
Our code borrow: 
- STGCN code from @
- C2FTCN code from @
- Metric code from @
Thanks for their efforts.
# Reference
If you found this repo useful, please cite this:

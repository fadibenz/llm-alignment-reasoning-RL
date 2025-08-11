# 2025-08-10:
- Initialize repository for the alignment project.
- Implement different SFT helper functions:
  - Function to tokenize prompt and output separately and then concatenates.
  - Function to calculate per-token entropy
  - Function to calculate response log-probabilities
  - Function to calculate normalized tensors with mask
- Implemented microbatch training step.
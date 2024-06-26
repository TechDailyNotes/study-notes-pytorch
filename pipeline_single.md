# PyTorch DNN Pipeline

1. Hyperparameter Setup
   1. Device Params
      1. `device`
   2. Data Params
      1. `batch_size`
   3. Model Params
      1. `input_size`
      2. `hidden_size`
      3. `num_classes`
      4. `lr`
   4. Training Params
      1. `num_epochs`
   5. Param Tuning
2. Data Setup
   1. Data Source
   2. Data Preprocessing
   3. Data Formatting
   4. Data Feature
   5. Data Accessories
3. Model Setup
   1. Architecture Setup
      1. Module Setup
      2. Module Connection
   2. Loss Setup
   3. Optimizer Setup
   4. Scheduler Setup
4. Training Loop
   1. Training Setup
      1. Mode Setup
      2. Device Setup
      3. Grad Setup
   2. Training Pass
      1. Forward Pass
      2. Loss Calculation
      3. Backward Pass
      4. Optimizer Step
      5. Scheduler Step
5. Result Test / Result Plot / Result Inference

## Data Pipeline

### Pixel Data

1. Images
   1. D-Type
      1. ToPILImage
      2. ToTensor
   2. Size
      1. Resize
      2. RandomCrop
   3. Orientation
      1. RandomRotation
      2. RandomHorizontalFlip
      3. RandomVerticalFlip
   4. Color
      1. RandomGrayscale
      2. ColorJitter
   5. Distribution
      1. Normalize

### Sequence Data

1. Text
   1. Tokenize
   2. Numericalize
      1. Index
      2. Embed
   3. Pad

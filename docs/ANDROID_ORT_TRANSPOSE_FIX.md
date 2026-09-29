# Android ORT Transpose regression

The first published 512 model pack failed on Android while loading
`vae_encoder.onnx`:

`Transpose TypeInferenceError: Invalid attribute perm {0,-1,1,2,3}`

The ONNX checker accepted that graph, but ONNX Runtime did not. The exporter
now rewrites negative Transpose axes to equivalent positive axes (for rank 5,
`-1` becomes `4`) and rejects any resulting value that is not a complete
permutation of `0..rank-1`.

The model-pack CI also creates a CPU ONNX Runtime `InferenceSession` for all
three exported graphs. Packaging is allowed only after both
`onnx.checker.check_model` and real ORT graph loading succeed.

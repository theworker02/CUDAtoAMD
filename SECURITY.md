# Security

Analyzer input is untrusted source. The analyzer reads only recognized source and binary artifact names, does not execute project code, and ignores `.git` contents. Future translators must treat generated kernels, binary inputs, caches, and remote protocol messages as untrusted. Remote execution is not implemented and must be explicit opt-in if introduced.

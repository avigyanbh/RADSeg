# Default Configurations
This directory includes all default configurations for the repo. You should override these configs with your own config or command line arguments as opposed to editing them.

We use the [hydra](https://hydra.cc/docs/intro/) configuration system which has a bit of a learning curve so getting familiar with hydra is encouraged but not necessary.

**TL;DR**:
- default.yaml is the config entry point and contains general configurations.
- Every directory specifies different options for a component that you can choose from in default.yaml.
- Every directory has a base config that all child components append.

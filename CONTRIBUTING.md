# Submitting an issue
Provide as much information as possible about your problem. No template for now.

# Contributing to the Codebase
## Code Structure

## Design Principles
Needless to say but always follow good software engineering and object oriented programming principles. The google style guide can help with some concepts. Other than that, here are some project specific principles that we follow:

1. Leverage bulk vectorized GPU operations as much as possible. Avoid loops when possible. 
2. Use PyTorch as a familiar API to manipulate data with CUDA.
3. Most data is in PyTorch format and we should eliminate conversions to other formats if possible.
4. Always support batched inputs/parameters when possible for better efficiency.
5. Don't reinvent the wheel UNLESS you will save us a troublesome dependency. Rule of thumb: if the dependency is not well maintained, or if the logic you need from it is trivial, consider writing the logic you need yourself.
6. (Adoption in progress) Be mindful of memory constraints. Doing everything in bulk can be compute efficient but can incur heavy memory consumption. Consider looping over chunks to break down large data dimensions (e.g looping over 100K voxels at a time where 100K should be dynamic or configurable).

## Code Formatting & Documentation
We follow the published [Google python coding style guide](https://google.github.io/styleguide/pyguide.html).
PRs not following this format will not be merged.

## Logging
Never use print statements to print out information to the user. Always use a logger.
Best practice is to define your logger at the top of your module like so:

```
import logging
logger = logging.getLogger(__name__)
```

Keep this excerpt in mind from the google style guide
```
For logging functions that expect a pattern-string (with %-placeholders) as their first argument: Always call them with a string literal (not an f-string!) as their first argument with pattern-parameters as subsequent arguments. Some logging implementations collect the unexpanded pattern-string as a queryable field. It also prevents spending time rendering a message that no logger is configured to output.
```

Follow proper severity levels when logging. More information can be found in the logging level tables [here](https://docs.python.org/3/howto/logging.html#when-to-use-logging).
## dust + gpu

Not a real package, just here for our next generation of dust + gpu support.

## Basic approach

The basic problem is that we want to keep a bunch of copies of state handy, and it's a bit tedious. We can divide the required state into a few basic bits:

* `y`: model state, minimally an array of length `n_state * n_particles`, but we could model this a number of ways
* `y_next`: the next model state, same shape as `y`
* `rng_state`: the random number state - `4 * uint64_t * n_particles`
* `model_internals`: this is not really in dust at the moment, but comes along with the definition of a model. Rather than storing a class of nice variables as we currently do we could store a massive block of memory and index into that with an offset - this will have size `len * n_particles` where `len` is something unknown as of yet. We need to decide if we'd be storing things as doubles or a block of memory and casting

Then we need to store all of that on the GPU too, so we'll get an extra copy. In the existing code this is stored as things like `_y_device`, and these are created with `cudaMalloc`, filled with `cudaMemcpy` and freed with `cudaFree`.

The interface so far is that we have a Particle class that holds gives us some level of abstraction - we can do that with this new version if it takes 4 pointers into these object. However, we need the underlying model object to be a bit different as I'm assuming that we can pass in an object with temporary storage and that's just not going to work.

All these blocks of memory should be interleaved together, as we believe that will lead to better performance.

## Interface issues

We'll have *every* model contain a CPU portion - that will take care of the parameter wrangling etc. But some models can then be made GPU aware, so we can offload their calculation onto the GPU if we want to. This means that we can keep dust relatively unchanged and continue to use nice stl types for most things.

This also massively simplifies the packaging as CPU and GPU versions can coexist in the same package easily, and we don't have to choose what we're generating.

## What is needed

Once a model is initialised on the CPU we need to get it to tell us:

* How long is `y` (and `y_state`)
* How long is `model_internals` if it was arranged as an array of `real_t`
* Convert from an initialised model to an interleaved array of data for each of these types
* For odin.dust, a version of the `update` method that works with this data type

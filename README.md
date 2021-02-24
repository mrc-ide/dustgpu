## dust + gpu

Not a real package, just here for our next generation of dust + gpu support.

Based on dust@56b0e022600869cf4b60f2969e60fb2a671c72e4
## Profiling

Setup (once):

- Ensure in src/Makevars `-pg --generate-line-info` is set in
  `NVCC_FLAGS`.
- Install R (>=v3.6)
- Install R packages by launching `/usr/bin/R` and running
  `install.packages(c("devtools", "socialmixr"))`

To run nsight systems profile:
```
nsys profile -o sirs_timeline_<commit> --trace cuda,osrt,openmp -c cudaProfilerApi --force-overwrite true /usr/bin/Rscript run_dust.R
```
This runs a full example where the kernels are called in sequence in
a loop. The profile runs from when the model object is created (`dustgpu::sir$new`)
to when its destructor is called (end of file).

To run nsight compute profile:
```
ncu -c 7 -o sirs_profile_<commit> --set full --target-processes all /usr/bin/Rscript run_dust.R
```
This is the same as above, but only a the first two kernel calls are profiled from
the loop. The parameters are set so that the first kernel uses more binomial inversion, the second more BTRS.

NB: The R files will compile the code for you if it has changed, as long as `nvcc` is on the path.

## Testing

Launch an R session, and from the R command line run:
```
pkgload::load_all()
devtools::test()
```

## The example

There is an example `inst/odin/sir.R` from which we generated `inst/dust/sir.cpp` and `src/sir.cpp` (`315d16e`)

Because we need to make changes to dust's include files, I've duplicated them into `inst/dust` and adjusted the search paths (`1cc7017`)

I'm then working against this commit to add the new features.

At `04fc091` we can extract the strided model internal state (this is exposed out as far as the R inteface but there's no need to do that at all).

Added R->S transitions by hand in `05b204`, with fixed rate 0.1.

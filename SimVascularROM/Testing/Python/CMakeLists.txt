# The package's own tests. They need neither Slicer nor a mesher: `svromsetup.testing` builds the
# vessel they run on, so they are a plain pytest run outside the application too. The ones that
# need svromutils or svZeroDSolver skip themselves where those are not to be found.

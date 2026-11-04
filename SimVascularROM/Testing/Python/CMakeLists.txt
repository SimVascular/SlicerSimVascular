# The package's own tests. They need neither Slicer nor a mesher: the whole case runs on
# svReducedOrderModelingUtils' testing aorta, which `svromsetup.testing_data` downloads and reads,
# so they are a plain pytest run outside the application too. The ones that need svromutils,
# svZeroDSolver or the aorta skip themselves where those are not to be had.

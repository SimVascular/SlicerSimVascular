# The package's own tests. They need neither Slicer nor a mesher: svmeshcomplete's cube is the
# mesh they write cases for, so they are a plain pytest run outside the application too. The one
# that runs svMultiPhysics skips itself where svMultiPhysics is not to be found.

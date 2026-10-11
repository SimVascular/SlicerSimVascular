#-----------------------------------------------------------------------------
set(MODULE_NAME SimVascularROM)

#-----------------------------------------------------------------------------
# `svromsetup` does the work; this module is an MRML adapter over it. Listing the package's files
# here installs them beside the module, which is already on Slicer's Python path, as Mesh Prep
# does with `svmeshcomplete`. `svromutils` is not listed: it is a package of its own, on PyPI,
# which the panel installs into Slicer's Python the first time it is needed.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  # The faces and the conditions table SimVascular MultiPhysics shares with this panel.
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/BoundaryConditionsTable.py
  # The 0D network drawn as a graph.
  ${MODULE_NAME}Lib/NetworkView.py
  svromsetup/__init__.py
  svromsetup/boundary_conditions.py
  svromsetup/case.py
  svromsetup/network.py
  svromsetup/results.py
  svromsetup/solver.py
  svromsetup/testing.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/UI/${MODULE_NAME}.ui
  LICENSE.txt
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()

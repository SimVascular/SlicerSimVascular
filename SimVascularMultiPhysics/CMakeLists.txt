#-----------------------------------------------------------------------------
set(MODULE_NAME SimVascularMultiPhysics)

#-----------------------------------------------------------------------------
# `svmpsetup` does the work; this module is an MRML adapter over it. Listing the package's files
# here installs them beside the module, which is already on Slicer's Python path, as Mesh Prep
# does with `svmeshcomplete`. The conditions and the table they are set in are SimVascular ROM
# Simulation's, imported from beside that module, which this one depends on.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  svmpsetup/__init__.py
  svmpsetup/case.py
  svmpsetup/settings.py
  svmpsetup/solver_xml.py
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

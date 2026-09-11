#-----------------------------------------------------------------------------
set(MODULE_NAME MeshCompleteExport)

#-----------------------------------------------------------------------------
# `svmeshcomplete` does the work; this module is an MRML adapter over it. Listing the
# package's files here installs them beside the module, which is already on Slicer's Python
# path -- so neither a binary install nor a development checkout on Additional Module Paths
# needs pip. `MeshCompleteExport/pyproject.toml` covers the other direction: importing it
# outside Slicer.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  svmeshcomplete/__init__.py
  svmeshcomplete/cells.py
  svmeshcomplete/face_table.py
  svmeshcomplete/faces.py
  svmeshcomplete/io.py
  svmeshcomplete/mesh_complete.py
  svmeshcomplete/testing.py
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

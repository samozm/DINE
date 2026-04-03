.onAttach <- function(libname, pkgname) {
  if (!check_openmp()) {
    packageStartupMessage("Note: DCENt was compiled without OpenMP. Multi-threading is disabled.")
  }
}
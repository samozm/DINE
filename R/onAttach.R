.onAttach <- function(libname, pkgname) {
  if (!check_openmp()) {
    packageStartupMessage("Note: DINE was compiled without OpenMP. Multi-threading is disabled.")
  }
}
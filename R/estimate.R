#' @export
estimate <- function(X,y,Z,n0,k0,t0,max_itr=200,covtype="",idx=1)
{
  startTime <- proc.time()
  
  res <- estimate_DEbeta(X,y,Z,n0,k0,t0,max_itr,covtype,idx) #a2.estimate_DEbeta(X,y,Z,n0,k0,t0,max_itr,covtype,idx)
  converged <- res$converged
  n_iter <- res$n_iter
  Sigma <- res$Sigma
  D <- res$D 
  E <- res$E 
  sigma <- res$sigma
  beta <- res$beta
  n_iter <- res$n_iter
  
  exeTimeClass <- proc.time() - startTime
  exeTime <- as.numeric(exeTimeClass[3])
  timelength <- exeTime
  #V <- Matrix::bdiag(Sigma)
  return(list(E=E,D=D,beta=beta,time=timelength,converged=converged,sigma=sigma,n_iter=n_iter))#,Sigma=V))
}
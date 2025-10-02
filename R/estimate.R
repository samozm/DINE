#' @export
estimate <- function(X,y,Z,n0,k0,t0,MAP,algo=2,max_itr=200,covtype="")
{
  startTime <- proc.time()
  nkt = 0
  for(i in 1:length(Z))
  {
    nkt = nkt + dim(Z[i])[1]
  }
  
  if(algo==2)
  {  
    res <- estimate_DEbeta(X,y,Z,MAP,n0,k0,t0,nkt,max_itr,covtype) #a2.estimate_DEbeta(X,y,Z,n0,k0,t0,max_itr,covtype,idx)
  }
  else if(algo==1)
  {
    res <- estimate_all(X,y,Z,n,k,t,max_itr,covtype)
  }
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
  return(list(E=E,D=D,V=Sigma,beta=beta,time=timelength,converged=converged,sigma=sigma,n_iter=n_iter))#,Sigma=V))
}
#' @export
estimate <- function(X,y,Z,n0,k0,t0,algo=2,max_itr=200,convergence_cutoff=5*(10^(-5)),REML=FALSE,verbose=FALSE,n_fold=5,threshold=NA)
{
  startTime <- proc.time()
  nkt = 0
  for(i in 1:length(Z))
  {
    nkt = nkt + dim(Z[i])[1]
  }
  V_nonzeros_pct = 0
  if(algo==2)
  {  
    custom_theta = F
    if(sum(is.na(threshold)) > 0)
    {
      threshold = matrix(0,2*k0,2*k0)
    } else {
      print("user inputted threshold, using custom theta")
      custom_theta = T
    }
    res <- estimate_DEbeta(X,y,Z,n0,k0,t0,threshold,max_itr,convergence_cutoff,REML,verbose,n_fold=n_fold,custom_theta=custom_theta) 
    #a2.estimate_DEbeta(X,y,Z,n0,k0,t0,max_itr,covtype,idx)
    sigma <- res$sigma
  }
  else if(algo==1)
  {
    res <- estimate_all(X,y,Z,n0,k0,t0,max_itr,convergence_cutoff,REML)
    sigma <- 0
    V_nonzeros_pct <- res$V_nonzeros_pct
  }
  converged <- res$converged
  n_iter <- res$n_iter
  Sigma <- res$Sigma
  D <- res$D 
  E <- res$E 
  
  beta <- res$beta
  
  n_iter <- res$n_iter
  
  exeTimeClass <- proc.time() - startTime
  exeTime <- as.numeric(exeTimeClass[3])
  timelength <- exeTime
  #V <- Matrix::bdiag(Sigma)
  return(list(E=E,D=D,V=Sigma,beta=beta,time=timelength,
              converged=converged,sigma=sigma,n_iter=n_iter,
              all_err=res$all_err,MAP=res$MAP,
              V_nonzeros_pct=V_nonzeros_pct,
              threshold=res$threshold))#,Sigma=V))
}

## Implementation of stochastic gradient descent

## install.packages("caret")
library(caret)
rm(list=ls())
lm_sgd = function(x, y, beta_init = NULL, learn_rate = 0.05, 
                  niter = 100, folds=10, verbose = F)
{
        
        n = nrow(x)
        p = ncol(x)-1
        
        if(nrow(x) != length(y)) stop("Check the dimensions of x and y")
        if(nrow(x) < folds) stop("folds exceed number of observations")
        if(verbose && p>2) stop("p > 2, -- Plotting not implemented")
        
        if(is.null(beta_init)) beta_init = runif(p+1)
        
        beta_sgd = beta_init
        
        f <- createFolds(y, k = folds, list = T, returnTrain = F)
        loss_sgd <- rep(0, niter)
        
        for(it in 1:niter)
        {
                for(batch in 1:folds){
                        x_b <- x[f[[batch]], ]
                        y_b <- y[f[[batch]]]
                        
                        beta_sgd = beta_sgd - learn_rate*(-2/length(y_b))*(t(x_b)%*%(y_b - x_b%*%beta_sgd))
                        
                        MSE_new = mean((y_b - x_b%*%beta_sgd)^2)
                        
                        if(verbose)
                        {
                                print(MSE_new)
                                
                                x1_grid = seq(0,1,length.out = 100)
                                y_grid_hat = cbind(rep(1,100),x1_grid)%*%beta_gd
                                
                                plot(x[,2],y)
                                lines(x1_grid,y_grid_hat)
                                Sys.sleep(0.1)
                        }
                }
                loss_sgd[it] <- MSE_new
        }
        return(list(beta_sgd, loss_sgd))
}

## Test
n = 30
p = 1

beta = rep(1,p+1)

x = cbind(rep(1,n),matrix(runif(n*p,0,1),n,p))
epsilon = rnorm(n,0,.1)

y = x%*%beta + epsilon 

lm_sgd(x, y)

## Implement stochastic gradient descent function
for(it in 1:niter)
{
        beta_gd = beta_gd - learn_rate*(-2/n)*(t(x)%*%(y-x%*%beta_gd))
        
        MSE_new = mean((y - x%*%beta_gd)^2)
        
        if(verbose)
        {
                print(MSE_new)
                
                x1_grid = seq(0,1,length.out = 100)
                y_grid_hat =  cbind(rep(1,100),x1_grid)%*%beta_gd
                
                plot(x[,2],y)
                lines(x1_grid,y_grid_hat)
                Sys.sleep(0.1)
        }
        loss_gd[it] <- MSE_new
}

## Implementation: linear regression
lm_gd = function(x, y, beta_init = NULL, learn_rate = 0.05, 
                 niter = 100, verbose = F)
{
        
        n = nrow(x)
        p = ncol(x)-1
        
        if(nrow(x) != length(y)) stop("Check the dimensions of x and y")
        if(verbose && p>2) stop("p > 2, -- Plotting not implemented")
        
        if(is.null(beta_init)) beta_init = runif(p+1)
        
        beta_gd = beta_init
        
        loss_gd <- rep(0, niter)
        
        for(it in 1:niter)
        {
                beta_gd = beta_gd - learn_rate*(-2/n)*(t(x)%*%(y-x%*%beta_gd))
                
                MSE_new = mean((y - x%*%beta_gd)^2)
                
                if(verbose)
                {
                        print(MSE_new)
                        
                        x1_grid = seq(0,1,length.out = 100)
                        y_grid_hat =  cbind(rep(1,100),x1_grid)%*%beta_gd
                        
                        plot(x[,2],y)
                        lines(x1_grid,y_grid_hat)
                        Sys.sleep(0.1)
                }
                loss_gd[it] <- MSE_new
        }
        return(list(beta_gd, loss_gd))
}


## Test
n = 30
p = 1

beta = rep(1,p+1)

x = cbind(rep(1,n),matrix(runif(n*p,0,1),n,p))
epsilon = rnorm(n,0,.1)

y = x%*%beta + epsilon 

## Store value of loss function
gd_loss <- lm_gd(x, y)[[2]]
sgd_loss <- lm_sgd(x, y)[[2]]

## Plot scatterplots of # iteration vs loss function for both gradient descent and stochastic gradient descent
par(mfrow=c(1,2))
yrange <- range(gd_loss, sgd_loss)
plot(x=1:100, y=sgd_loss, ylim = yrange, 
     xlab = "# iterations", ylab = "lm_sgd loss function")
plot(x=1:100, y=gd_loss, ylim = yrange, 
     xlab = "# iterations", ylab = "lm_gd loss function")


library(purrr)
## Generate a list of 20 random vectors
beta_init <- map(1:20, ~ sample(1:100, size=2))

## Run stochastic gradient descent with initialization value 
## for every element in the list
## Display estimation errors
map(beta_init, ~ lm_sgd(x,y)[[1]][[2]])
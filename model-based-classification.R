
data(vaso, package='robustbase')

d0 <- subset(vaso, Y == 0, select = c('Volume', 'Rate')) 
d1 <- subset(vaso, Y == 1, select = c('Volume', 'Rate')) 

( mu0 <- colMeans( d0 ) )
( mu1 <- colMeans( d1 ) )

( sigma0 <- cov( d0 ) )
( sigma1 <- cov( d1 ) )

(sigma.hat <- ( (nrow(d0) - 1 ) * sigma0 
                + (nrow(d1) - 1) * sigma1 ) / (nrow(vaso) - 2) )

( pi0 <- with(vaso, mean(Y == 0)) )
( pi1 <- with(vaso, mean(Y == 1)) )

x0 <- c(1.9, 1.3)
p <- 2
cp0 <- as.numeric( crossprod( x0 - mu0, solve(sigma.hat, x0 - mu0)) )
dens0 <- det(sigma.hat)^(-1/2) * exp( - cp0 / 2 ) * (sqrt(2*pi)^(-p))
cp1 <- as.numeric( crossprod( x0 - mu1, solve(sigma.hat, x0 - mu1)) )
dens1 <- det(sigma.hat)^(-1/2) * exp( - cp1 / 2 ) * (sqrt(2*pi)^(-p))

( pp0 <- dens0 * pi0 )
( pp1 <- dens1 * pi1 )

c( posterior0 = pp0 / (pp0 + pp1), posterior1 = pp1 / (pp0 + pp1) )

dens0 <- mvtnorm::dmvnorm(x0, mean=mu0, sigma=sigma.hat)
dens1 <- mvtnorm::dmvnorm(x0, mean=mu1, sigma=sigma.hat)

( pp0 <- dens0 * pi0 )
( pp1 <- dens1 * pi1 )
c( posterior0 = pp0 / (pp0 + pp1), posterior1 = pp1 / (pp0 + pp1) )

library(MASS)
a.lda <- lda(Y ~ Volume + Rate, data=vaso)
names(x0) <- c('Volume', 'Rate')
predict(a.lda, newdata=data.frame(t(x0)))$posterior


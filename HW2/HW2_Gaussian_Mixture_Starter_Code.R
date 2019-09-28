# Generating Gaussian Mixture)

p_head = 0.7                  #Prob. of heads
N = 100                       #Total Draws

#Simulate a Biased Coin Flip (The Bias represents the prior probability pi1, pi0)
flip <- function(p_head){
  return(ifelse(runif(1) <= (1 - p_head),0,1))
}

#Write out a Vector of results of N coin flips
flips <- c()
i=1
while(i<=100) {
    flips <- c(flips,flip(p_head))
    i=i+1
}
#Check you have around (1-p_head)% tails
#sum(flips == 0)

#Following Function samples from two Normal Distribution on the basis of a coin toss
draw <- function(coin){
  if(coin == 1){
  return(rnorm(1, mean=1.5, sd=1)) #Mean 1.5, SD 1
  }
else{
  return(rnorm(1, mean=0, sd=1)) #Mean 0, SD 1
}}

#Using the sequence of flips alread generated we will create a gaussian mixture Vector

mixture <- c()
j=1
while(j<=100) {
  mixture <- c(mixture,draw(flips[j]))
  j=j+1
}

# Bind mixture with flips as a Dataframe. This dataframe can be directly used with the
#classification algorithms

mix <- do.call(rbind, Map(data.frame, X=mixture, Y=flips))
